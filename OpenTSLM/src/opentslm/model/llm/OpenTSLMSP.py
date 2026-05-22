# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from opentslm.model_config import ENCODER_OUTPUT_DIM
from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from ..projector.MLPProjector import MLPProjector
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


class OpenTSLMSP(TimeSeriesLLM):
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
    ):
        super().__init__(device)

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) encoder + projector (now internal)
        self.encoder = TransformerCNNEncoder().to(device)
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)

        self.patch_size = 4

        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = (
            None  # Keep reference to original model for backward compatibility
        )

        # Optional scalar-regression head (v1b "hybrid" variant for MISATO).
        # Pools the LLM's last non-pad input-position hidden state and maps it
        # to a single scalar (e.g. pK). When enabled, compute_loss adds
        # `regression_weight * MSE(pred, batch[i]["pK"])` to the LM loss.
        self.regression_enabled: bool = False
        self.regression_weight: float = 0.0
        self.regression_head: Optional[nn.Module] = None

        # Optional multi-task heads (v2 stack). Each reads the same pooled
        # hidden state and produces an auxiliary supervision signal. Weights
        # of 0 disable the corresponding loss term.
        self.multitask_enabled: bool = False
        self.dissoc_head: Optional[nn.Module] = None
        self.drift_head: Optional[nn.Module] = None
        self.aux_reg_head: Optional[nn.Module] = None
        self.multitask_weights: Dict[str, float] = {
            "dissoc": 0.0, "drift": 0.0, "aux_reg": 0.0,
        }

        # Optional pair-ranking loss on the regression head's scalar output.
        # Targets Pearson R rather than absolute pK. Margin in pK units.
        self.ranking_weight: float = 0.0
        self.ranking_margin: float = 0.5

        # Freeze the LLM backbone for SP model (internally)
        for p in self.llm.parameters():
            p.requires_grad = False

    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Enable LoRA fine-tuning for the LLM component.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to. If None, uses defaults.
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine-tuning. Please install with: pip install peft"
            )

        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first if you want to reconfigure LoRA."
            )

        # Store reference to original model before applying LoRA
        self.original_llm = self.llm

        # Default target modules for common architectures
        if target_modules is None:
            target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            # Apply LoRA to the model
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True

            # Print LoRA info
            lora_params = sum(
                p.numel()
                for name, p in self.llm.named_parameters()
                if p.requires_grad and "lora_" in name
            )
            trainable_params = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA enabled:")
            print(f"   LoRA parameters: {lora_params:,}")
            print(f"   Total trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   LoRA %: {100 * lora_params / total_params:.2f}%")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            print(f"❌ Failed to enable LoRA: {e}")
            print(
                "   This might be due to incompatible target modules for your model architecture."
            )
            print(
                "   Try specifying different target_modules or check your model's layer names."
            )
            raise

    def get_lora_parameters(self):
        """Get LoRA parameters for the optimizer."""
        if not self.lora_enabled:
            return []

        lora_params = []
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params.append(param)
        return lora_params

    def enable_regression(self, weight: float = 0.5, hidden_dim: int = 512, dropout: float = 0.1):
        """Attach a 2-layer MLP regression head on the LLM's last input-position hidden state.

        After `enable_regression`, `compute_loss` returns
        `L_LM + weight * MSE(head(pooled), batch["pK"])` whenever batch samples
        carry a `pK` field. `predict_pK` returns scalar predictions for
        inference.
        """
        if self.regression_enabled:
            raise RuntimeError("regression head already enabled")
        H = self.llm.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(H, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.regression_weight = float(weight)
        self.regression_enabled = True
        n_params = sum(p.numel() for p in self.regression_head.parameters())
        print(f"✅ regression head enabled: {n_params:,} params, weight={weight}")

    def get_regression_parameters(self) -> List[nn.Parameter]:
        if not self.regression_enabled or self.regression_head is None:
            return []
        return list(self.regression_head.parameters())

    def enable_multitask(
        self,
        dissoc_weight: float = 0.1,
        drift_weight: float = 0.1,
        aux_reg_weight: float = 0.05,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Attach binary + scalar auxiliary heads on the pooled hidden state.

        Heads (each a small 2-layer MLP, ~150k params total):
          - dissoc_head  : sigmoid → BCE on batch["dissociated"]
          - drift_head   : sigmoid → BCE on batch["ligand_drift"]
          - aux_reg_head : scalar  → MSE on batch["bsasa_drift"] (in raw units)

        compute_loss adds `Σ_k weight_k · L_k` once enabled, but only for
        batches whose samples carry the corresponding fields. Setting a
        weight to 0 disables that loss term while keeping the head's
        parameters in the checkpoint.
        """
        if self.multitask_enabled:
            raise RuntimeError("multitask heads already enabled")
        H = self.llm.config.hidden_size

        def _head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(H, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.dissoc_head = _head(1).to(self.device)
        self.drift_head = _head(1).to(self.device)
        self.aux_reg_head = _head(1).to(self.device)
        self.multitask_weights = {
            "dissoc": float(dissoc_weight),
            "drift": float(drift_weight),
            "aux_reg": float(aux_reg_weight),
        }
        self.multitask_enabled = True
        n_params = sum(p.numel() for m in (self.dissoc_head, self.drift_head,
                                            self.aux_reg_head)
                       for p in m.parameters())
        print(f"✅ multitask heads enabled: {n_params:,} params, "
              f"weights={self.multitask_weights}")

    def get_multitask_parameters(self) -> List[nn.Parameter]:
        if not self.multitask_enabled:
            return []
        params: List[nn.Parameter] = []
        for h in (self.dissoc_head, self.drift_head, self.aux_reg_head):
            if h is not None:
                params.extend(h.parameters())
        return params

    def enable_ranking_loss(self, weight: float = 0.2, margin: float = 0.5) -> None:
        """Activate margin-based pair ranking loss on the regression head's pK.

        Requires `enable_regression()` to have been called — the head's scalar
        output is the quantity we rank. Pairs are sampled in-batch (B-1 pairs
        from a random permutation, no self-pair). Adds
            weight · Σ_pairs max(0, margin − sign(pK_i−pK_j)·(h_i−h_j))
        to the total loss. Margin in pK units.
        """
        if not self.regression_enabled:
            raise RuntimeError(
                "Ranking loss needs a regression head; call enable_regression() first."
            )
        self.ranking_weight = float(weight)
        self.ranking_margin = float(margin)
        print(f"✅ ranking loss enabled: weight={weight}, margin={margin} pK")

    def disable_lora(self):
        """Disable LoRA and revert to original frozen LLM."""
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is not enabled. Cannot disable LoRA when it's not active."
            )

        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None

        self.lora_enabled = False
        print("✅ LoRA disabled, reverted to frozen LLM")

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TL;DR:
            This function is probably the most crucial part of OpenTSLM-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
        """
        device = self.device
        H = self.llm.config.hidden_size

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N_all, P_max, H]

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                # ensure [T] → [T,1]
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            # ── pad time dim to multiple of patch_size ──
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)
            # ── now ts_padded: [N_ts_total, T_padded, 1]

            # ── key fix: squeeze out the feature dim so encoder sees [B, L] ──
            ts_enc = self.encoder(
                ts_padded.squeeze(-1)
            )  # [N_ts_total, N_patches, embed_dim]
            ts_proj = self.projector(ts_enc).to(
                text_embeds.dtype
            )  # [N_ts_total, N_patches, H]
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) Re‐assemble per sample
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]  # [1+N_ts+1, P_max, H]
            sample_masks = attn_mask[start:end]  # [1+N_ts+1, P_max]
            seq_embeds, seq_masks = [], []

            # pre_prompt
            length = sample_masks[0].sum().item()
            seq_embeds.append(sample_embeds[0, :length, :])
            seq_masks.append(sample_masks[0, :length])

            # each (textᵢ, tsᵢ)
            for i in range(n_ts):
                idx = 1 + i
                length = sample_masks[idx].sum().item()
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(sample_masks[idx, :length])

                proj = ts_proj[ts_offset + i]  # [N_patches, H]
                seq_embeds.append(proj)
                seq_masks.append(
                    torch.ones(proj.size(0), device=device, dtype=torch.long)
                )

            ts_offset += n_ts

            # post_prompt (fixed)
            length = sample_masks[-1].sum().item()
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(sample_masks[-1, :length])

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))

        # 5) Batch-pad the final sequences
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        return inputs_embeds, attention_mask

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def compute_loss(self, batch: List[Dict[str, any]],
                     return_breakdown: bool = False):
        """Compute LM loss, optionally augmented with v1b regression, multi-task
        auxiliary heads, and a pair-ranking loss.

        Returns:
          - tensor `loss` if return_breakdown=False (back-compat with trainer)
          - (loss, dict) where dict has float values for each loss term, useful
            for W&B logging or sanity checks.

        The pooled hidden state used by all heads is the LLM's last non-pad
        INPUT position (not answer positions) — same as the v1b head.
        """
        answers = [b["answer"] for b in batch]

        inputs_embeds, input_attention_mask = self.pad_and_apply_batch(batch)
        B, L, H = inputs_embeds.size()

        # tokenize answers
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)  # [B, A_max, H]

        # append
        full_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)  # [B, L+A, H]
        full_mask = torch.cat([input_attention_mask, ans_mask], dim=1)  # [B, L+A]

        # labels: only on the answer tokens
        total_len = full_mask.size(1)
        labels = torch.full((B, total_len), -100, device=self.device, dtype=torch.long)
        labels[:, L:] = ans_ids

        run_regression = (
            self.regression_enabled
            and self.regression_head is not None
            and all("pK" in b for b in batch)
        )
        run_multitask = self.multitask_enabled and (
            self.dissoc_head is not None
            or self.drift_head is not None
            or self.aux_reg_head is not None
        )
        need_hidden = run_regression or run_multitask

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=need_hidden,
        )

        breakdown: Dict[str, float] = {"lm": float(outputs.loss.detach())}
        if not need_hidden:
            total = outputs.loss
            return (total, breakdown) if return_breakdown else total

        # Pool the last non-pad input position per sample.
        input_lengths = input_attention_mask.sum(dim=1).long()  # [B]
        last_idx = (input_lengths - 1).clamp(min=0)
        last_hidden = outputs.hidden_states[-1]  # [B, L+A, H]
        pooled = last_hidden[torch.arange(B, device=self.device), last_idx]

        total = outputs.loss

        # --- v1b regression head ---
        pK_pred = None
        if run_regression:
            head_dtype = self.regression_head[0].weight.dtype
            pK_pred = self.regression_head(pooled.to(head_dtype)).squeeze(-1)
            pK_true = torch.tensor(
                [b["pK"] for b in batch], device=self.device, dtype=pK_pred.dtype
            )
            mse = F.mse_loss(pK_pred, pK_true)
            total = total + self.regression_weight * mse
            breakdown["regression_mse"] = float(mse.detach())

        # --- ranking loss on the regression head's scalar output ---
        if (run_regression and self.ranking_weight > 0
                and pK_pred is not None and B >= 2):
            pK_true_t = torch.tensor(
                [b["pK"] for b in batch], device=self.device, dtype=pK_pred.dtype
            )
            # Use a random permutation to pair each sample with a different one.
            perm = torch.randperm(B, device=self.device)
            mask = perm != torch.arange(B, device=self.device)
            if mask.any():
                idx_i = torch.arange(B, device=self.device)[mask]
                idx_j = perm[mask]
                diff_pred = pK_pred[idx_i] - pK_pred[idx_j]
                diff_true = pK_true_t[idx_i] - pK_true_t[idx_j]
                sign = torch.sign(diff_true)
                rank = F.relu(self.ranking_margin - sign * diff_pred).mean()
                total = total + self.ranking_weight * rank
                breakdown["ranking"] = float(rank.detach())

        # --- multi-task auxiliary heads ---
        if run_multitask:
            head_dtype = (self.dissoc_head[0].weight.dtype
                          if self.dissoc_head is not None
                          else pooled.dtype)
            pooled_h = pooled.to(head_dtype)

            if (self.dissoc_head is not None and self.multitask_weights["dissoc"] > 0
                    and all("dissociated" in b for b in batch)):
                logits = self.dissoc_head(pooled_h).squeeze(-1)
                targets = torch.tensor(
                    [float(b["dissociated"]) for b in batch],
                    device=self.device, dtype=logits.dtype,
                )
                loss_d = F.binary_cross_entropy_with_logits(logits, targets)
                total = total + self.multitask_weights["dissoc"] * loss_d
                breakdown["aux_dissoc_bce"] = float(loss_d.detach())

            if (self.drift_head is not None and self.multitask_weights["drift"] > 0
                    and all("ligand_drift" in b for b in batch)):
                logits = self.drift_head(pooled_h).squeeze(-1)
                targets = torch.tensor(
                    [float(b["ligand_drift"]) for b in batch],
                    device=self.device, dtype=logits.dtype,
                )
                loss_dr = F.binary_cross_entropy_with_logits(logits, targets)
                total = total + self.multitask_weights["drift"] * loss_dr
                breakdown["aux_drift_bce"] = float(loss_dr.detach())

            if (self.aux_reg_head is not None and self.multitask_weights["aux_reg"] > 0
                    and all("bsasa_drift" in b for b in batch)):
                pred = self.aux_reg_head(pooled_h).squeeze(-1)
                targets = torch.tensor(
                    [b["bsasa_drift"] for b in batch],
                    device=self.device, dtype=pred.dtype,
                )
                # Rescale target by 100 (raw Å² range is ~[-500, +500]) so the
                # MSE magnitude is comparable to other losses.
                loss_a = F.mse_loss(pred, targets / 100.0)
                total = total + self.multitask_weights["aux_reg"] * loss_a
                breakdown["aux_bsasa_drift_mse"] = float(loss_a.detach())

        breakdown["total"] = float(total.detach())
        return (total, breakdown) if return_breakdown else total

    @torch.no_grad()
    def predict_pK(self, batch: List[Dict[str, any]]) -> List[float]:
        """Return scalar pK predictions from the regression head.

        Requires `enable_regression()` to have been called. Does not generate
        text — runs a single forward pass over inputs only.
        """
        if not self.regression_enabled or self.regression_head is None:
            raise RuntimeError("regression head is not enabled; call enable_regression() first")
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B = inputs_embeds.size(0)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        input_lengths = attention_mask.sum(dim=1).long()
        last_idx = (input_lengths - 1).clamp(min=0)
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[torch.arange(B, device=self.device), last_idx]
        pK_pred = self.regression_head(pooled.to(self.regression_head[0].weight.dtype)).squeeze(-1)
        return pK_pred.float().cpu().tolist()

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def store_to_file(self, path: str):
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
        }

        # Add LoRA state to checkpoint
        self.save_lora_state_to_checkpoint(checkpoint)

        # Add regression-head state (v1b)
        if self.regression_enabled and self.regression_head is not None:
            checkpoint["regression_enabled"] = True
            checkpoint["regression_weight"] = self.regression_weight
            checkpoint["regression_head_state"] = self.regression_head.state_dict()
            if self.ranking_weight > 0:
                checkpoint["ranking_weight"] = self.ranking_weight
                checkpoint["ranking_margin"] = self.ranking_margin

        # Add multi-task heads (v2 stack)
        if self.multitask_enabled:
            checkpoint["multitask_enabled"] = True
            checkpoint["multitask_weights"] = self.multitask_weights
            if self.dissoc_head is not None:
                checkpoint["dissoc_head_state"] = self.dissoc_head.state_dict()
            if self.drift_head is not None:
                checkpoint["drift_head_state"] = self.drift_head.state_dict()
            if self.aux_reg_head is not None:
                checkpoint["aux_reg_head_state"] = self.aux_reg_head.state_dict()

        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])

        # Load LoRA state if present (allow missing for backward compatibility)
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)

        # Load regression head if present (allow missing for back-compat)
        if ckpt.get("regression_enabled", False):
            if not self.regression_enabled:
                self.enable_regression(weight=ckpt.get("regression_weight", 0.5))
            self.regression_head.load_state_dict(ckpt["regression_head_state"])
            if "ranking_weight" in ckpt:
                self.ranking_weight = float(ckpt["ranking_weight"])
                self.ranking_margin = float(ckpt.get("ranking_margin", 0.5))
            print(f"📥 Loaded regression head (weight={self.regression_weight})")

        # Load multi-task heads if present
        if ckpt.get("multitask_enabled", False):
            mt_w = ckpt.get("multitask_weights",
                            {"dissoc": 0.1, "drift": 0.1, "aux_reg": 0.05})
            if not self.multitask_enabled:
                self.enable_multitask(
                    dissoc_weight=mt_w.get("dissoc", 0.1),
                    drift_weight=mt_w.get("drift", 0.1),
                    aux_reg_weight=mt_w.get("aux_reg", 0.05),
                )
            if "dissoc_head_state" in ckpt and self.dissoc_head is not None:
                self.dissoc_head.load_state_dict(ckpt["dissoc_head_state"])
            if "drift_head_state" in ckpt and self.drift_head is not None:
                self.drift_head.load_state_dict(ckpt["drift_head_state"])
            if "aux_reg_head_state" in ckpt and self.aux_reg_head is not None:
                self.aux_reg_head.load_state_dict(ckpt["aux_reg_head_state"])
            print(f"📥 Loaded multitask heads (weights={self.multitask_weights})")

        print(f"📥 Loaded model from epoch {ckpt.get('epoch', '?')}")

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """
        Load LoRA adapters from a checkpoint.

        Args:
            checkpoint: Checkpoint dictionary containing potential LoRA state
            allow_missing: If True, don't raise exception when checkpoint has no LoRA but model expects it

        Raises:
            RuntimeError: When there's a mismatch between checkpoint and current LoRA state
        """
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)

        if checkpoint_has_lora and "lora_state" in checkpoint:
            # Checkpoint has LoRA adapters
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )

            # Load LoRA adapters
            try:
                lora_state = checkpoint["lora_state"]
                loaded_count = 0
                missing_keys = []

                # Track which LoRA parameters we expect to find
                expected_lora_params = {
                    name
                    for name, param in self.llm.named_parameters()
                    if param.requires_grad and "lora_" in name
                }

                for name, param in self.llm.named_parameters():
                    if name in lora_state and param.requires_grad and "lora_" in name:
                        param.data.copy_(lora_state[name])
                        loaded_count += 1
                    elif param.requires_grad and "lora_" in name:
                        missing_keys.append(name)

                if missing_keys and not allow_missing:
                    raise RuntimeError(
                        f"Could not find LoRA parameters in checkpoint: {missing_keys[:5]}..."
                    )

                print(f"📥 Loaded LoRA adapters: {loaded_count} parameters")
                return loaded_count

            except Exception as e:
                if "Could not find LoRA parameters" in str(e):
                    raise  # Re-raise our custom exception
                raise RuntimeError(f"Failed to load LoRA adapters: {e}")

        elif checkpoint_has_lora:
            raise RuntimeError(
                "Checkpoint indicates LoRA was enabled but no LoRA state found"
            )

        # Handle case where checkpoint has no LoRA but model expects it
        if not checkpoint_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError(
                    "Loading checkpoint from before LoRA was enabled, but LoRA is currently enabled. "
                    "LoRA adapters will be randomly initialized. Set allow_missing=True to allow this."
                )
            else:
                print("⚠️  Loading checkpoint from before LoRA was enabled.")
                print("   LoRA adapters will be randomly initialized.")

        return 0

    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """
        Save LoRA adapters to a checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary to add LoRA state to

        Returns:
            int: Number of LoRA parameters saved
        """
        checkpoint["lora_enabled"] = self.lora_enabled

        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            try:
                # Save LoRA adapter weights
                lora_state = {}
                for name, param in self.llm.named_parameters():
                    if param.requires_grad and "lora_" in name:
                        lora_state[name] = param.data.clone()

                if lora_state:
                    checkpoint["lora_state"] = lora_state
                    checkpoint["lora_config"] = self.llm.peft_config
                    print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
                    return len(lora_state)
            except Exception as e:
                raise RuntimeError(f"Failed to save LoRA adapters: {e}")

        return 0

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 30000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """

        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, normalize=normalize
        )
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]
