"""Agent loop orchestrator — task #17.

Implements evaluate_agent() per FRONTEND_v2.md §9.2:
  pre-flight (lookup_split + rag_query)
  → loop (Claude via OpenRouter ⇄ tools, max 8 steps)
  → strict-JSON verdict (schema §9.4) with one corrective retry on parse error

The trace is structured for the UI: per-step {tool, input, result, latency_ms}.
RAG chunk text is stripped (snippet[:200]) so the wire payload stays small.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator

from llm import spend as spend_mod
from llm.openrouter import messages_create
from tools import TOOL_REGISTRY, TOOL_SCHEMAS


class TraceStep(TypedDict):
    step: int
    tool: str
    input: dict[str, Any]
    result: dict[str, Any]
    latency_ms: int


# --------------------------------------------------------------------------
# Verdict schema (matches §9.4)
# --------------------------------------------------------------------------


class VerdictScores(BaseModel):
    structural_consistency: float = Field(ge=0, le=1)
    physical_consistency: float = Field(ge=0, le=1)
    literature_consistency: float = Field(ge=0, le=1)
    chemical_plausibility: float = Field(ge=0, le=1)


class ClaimEvidence(BaseModel):
    claim: str
    evidence: str


class ClaimContradiction(BaseModel):
    claim: str
    contradicting_evidence: str


class MissingClaim(BaseModel):
    evidence: str
    why_relevant: str


class Citation(BaseModel):
    chunk_id: str
    score: float = Field(ge=0, le=2)  # post-rerank scores can exceed 1


class Verdict(BaseModel):
    scores: VerdictScores
    verified_claims: list[ClaimEvidence] = []
    contradicted_claims: list[ClaimContradiction] = []
    missing_claims: list[MissingClaim] = []
    recommendation: str
    citations: list[Citation] = []
    independence_caveats: list[str] = []

    @field_validator("recommendation")
    @classmethod
    def _valid_rec(cls, v: str) -> str:
        if v not in ("trust", "review", "discard"):
            raise ValueError("recommendation must be trust|review|discard")
        return v


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


_PROMPT_DIR = Path(__file__).parent / "prompts"


def _system_prompt() -> str:
    return (_PROMPT_DIR / "system.md").read_text()


def _user_prompt(
    *, pdb_id: str, variant: str, split: dict, model_pK: float,
    rationale: str, initial_rag: dict,
) -> str:
    tmpl = (_PROMPT_DIR / "user_template.md").read_text()
    return tmpl.format(
        pdb_id=pdb_id,
        variant=variant,
        split=split.get("split", "unknown"),
        model_pK=f"{model_pK:.2f}",
        rationale=rationale.strip() or "(empty rationale)",
        initial_rag=_format_rag_for_prompt(initial_rag),
    )


def _format_rag_for_prompt(rag: dict) -> str:
    chunks = rag.get("chunks", [])
    if not chunks:
        return "(no chunks retrieved)"
    lines = []
    for c in chunks[:6]:
        snippet = c.get("text", "")[:400].replace("\n", " ")
        lines.append(f"[{c['chunk_id']}] score={c['score']:.2f}\n  {snippet}")
    return "\n".join(lines)


def _strip_rag_for_trace(result: dict) -> dict:
    """rag_query results contain full chunk text — trim before sending to client."""
    if not isinstance(result, dict) or "chunks" not in result:
        return result
    chunks = result.get("chunks", [])
    return {
        **result,
        "chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
                "contains_label": c.get("contains_label"),
                "pdb_ids": c.get("pdb_ids"),
                "snippet": (c.get("text") or "")[:200],
                "source": c.get("source"),
            }
            for c in chunks
        ],
    }


def _extract_text(content: list[dict]) -> str:
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def _extract_last_json_object(text: str) -> str:
    """Find the last syntactically-valid top-level JSON object in `text`.

    Earlier versions used `text.rfind("{") + text.rfind("}")`, which broke
    when the verdict contained nested objects (the rfind picks the LAST
    `{`, which is usually the opening brace of an inner claim dict, not
    the outermost verdict). raw_decode handles balanced braces and
    string-escape edge cases for free.

    Also strips a ```json ... ``` markdown fence if present.
    """
    # 1. Markdown fence — if the model wrapped its JSON, peel the fence.
    fence_start = text.rfind("```json")
    if fence_start >= 0:
        rest = text[fence_start + len("```json"):]
        fence_end = rest.find("```")
        if fence_end >= 0:
            candidate = rest[:fence_end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass  # fall through to scan-based extraction

    # 2. Scan for the LAST valid top-level JSON object.
    decoder = json.JSONDecoder()
    last: str | None = None
    idx = 0
    while idx < len(text):
        open_at = text.find("{", idx)
        if open_at < 0:
            break
        try:
            _, end = decoder.raw_decode(text[open_at:])
            last = text[open_at:open_at + end]
            idx = open_at + end
        except json.JSONDecodeError:
            idx = open_at + 1
    if last is None:
        raise ValueError("no valid JSON object found in model output")
    return last


def _parse_verdict(text: str) -> Verdict:
    """Strict pydantic parse. Raises on missing fields or wrong shape."""
    obj = _extract_last_json_object(text)
    return Verdict.model_validate_json(obj)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


async def evaluate_agent(
    pdb_id: str,
    model_pK: float,
    rationale: str,
    variant: str,
    max_steps: int = 8,
) -> tuple[dict[str, Any], list[TraceStep]]:
    """Full audit loop. Returns (verdict_dict, trace)."""
    # ----- Pre-flight (always run, before LLM sees anything) -----
    split = TOOL_REGISTRY["lookup_split"](pdb_id=pdb_id)
    initial_rag = TOOL_REGISTRY["rag_query"](
        query=f"binding mode of {pdb_id}", pdb_id=pdb_id, top_k=6,
    )

    trace: list[TraceStep] = [
        {"step": -1, "tool": "lookup_split", "input": {"pdb_id": pdb_id},
         "result": split, "latency_ms": 0},
        {"step": -1, "tool": "rag_query",
         "input": {"query": f"binding mode of {pdb_id}", "pdb_id": pdb_id, "top_k": 6},
         "result": _strip_rag_for_trace(initial_rag), "latency_ms": 0},
    ]

    system = _system_prompt()
    user = _user_prompt(
        pdb_id=pdb_id, variant=variant, split=split,
        model_pK=model_pK, rationale=rationale, initial_rag=initial_rag,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

    total_in = total_out = total_cache = 0
    last_response = None

    # ----- LLM loop -----
    for step in range(max_steps):
        spend_mod.check_or_429()

        resp = await messages_create(
            system=system,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=2048,
            temperature=0.0,
        )
        last_response = resp
        total_in += resp["usage"]["input_tokens"]
        total_out += resp["usage"]["output_tokens"]
        total_cache += resp["usage"]["cache_read_input_tokens"]

        messages.append({"role": "assistant", "content": resp["content"]})

        if resp["stop_reason"] == "end_turn":
            break

        # Dispatch tool calls
        tool_results = []
        for block in resp["content"]:
            if block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            args = block.get("input", {}) or {}
            t0 = time.monotonic()
            try:
                fn = TOOL_REGISTRY.get(name)
                if fn is None:
                    result = {"error": f"unknown tool: {name}"}
                else:
                    result = fn(**args)
            except TypeError as e:
                result = {"error": f"bad args for {name}: {e}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            latency = int((time.monotonic() - t0) * 1000)
            trace.append({
                "step": step, "tool": name, "input": args,
                "result": _strip_rag_for_trace(result), "latency_ms": latency,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(result)[:8000],  # cap on long tool outputs
            })

        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})

    # ----- Parse verdict, with one corrective retry on schema fail -----
    text = _extract_text(last_response["content"]) if last_response else ""
    verdict: Verdict | None = None
    parse_error: str | None = None
    try:
        verdict = _parse_verdict(text)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        parse_error = str(e)

    if verdict is None:
        # One corrective turn
        messages.append({
            "role": "user",
            "content": (
                f"Your last response did not validate against the verdict schema. "
                f"Error: {parse_error}\n\n"
                "Respond NOW with only a single JSON object matching the schema in "
                "the system prompt, no surrounding prose."
            ),
        })
        try:
            resp2 = await messages_create(
                system=system, messages=messages, tools=[], max_tokens=2048, temperature=0.0,
            )
            total_in += resp2["usage"]["input_tokens"]
            total_out += resp2["usage"]["output_tokens"]
            text2 = _extract_text(resp2["content"])
            verdict = _parse_verdict(text2)
        except Exception as e:
            # Graceful fallback per task #17 note: never 500.
            # _degraded=True tells app.py to skip caching so the user gets a
            # fresh attempt on the next click instead of a permanent failure.
            verdict_dict = {
                "scores": {
                    "structural_consistency": 0.0,
                    "physical_consistency": 0.0,
                    "literature_consistency": 0.0,
                    "chemical_plausibility": 0.0,
                },
                "verified_claims": [],
                "contradicted_claims": [],
                "missing_claims": [],
                "recommendation": "review",
                "citations": [],
                "independence_caveats": [f"judge produced malformed output: {e}"],
                "_degraded": True,
            }
            return _attach_envelope(verdict_dict, trace, total_in, total_out, total_cache), trace

    verdict_dict = verdict.model_dump()
    return _attach_envelope(verdict_dict, trace, total_in, total_out, total_cache), trace


def _attach_envelope(
    verdict_dict: dict, trace: list[TraceStep],
    total_in: int, total_out: int, total_cache: int,
) -> dict:
    """Wrap verdict with the §9.4 envelope keys the frontend expects."""
    total_latency = sum(s["latency_ms"] for s in trace)
    verdict_dict.update({
        "judge_model": os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7"),
        "rag_corpus_version": os.getenv("RAG_CORPUS_VERSION", "v1-unset"),
        "agent_trace": {
            "tool_calls": sum(1 for s in trace if s["step"] >= 0),
            "latency_ms": total_latency,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cache_read_input_tokens": total_cache,
        },
    })
    return verdict_dict


async def evaluate_fast(
    pdb_id: str,
    model_pK: float,
    rationale: str,
    variant: str,
) -> dict[str, Any]:
    """RAG + single LLM call, no tool loop — task #18 /evaluate endpoint.

    ~3-5 s, ~$0.02. Same verdict schema, but agent_trace.tool_calls=0 so the
    UI can render an abbreviated audit panel.
    """
    split = TOOL_REGISTRY["lookup_split"](pdb_id=pdb_id)
    initial_rag = TOOL_REGISTRY["rag_query"](
        query=f"binding mode of {pdb_id}", pdb_id=pdb_id, top_k=8,
    )

    system = _system_prompt() + (
        "\n\nFAST MODE: No additional tools available. Respond with ONLY the "
        "JSON verdict in one turn — no plan, no tool calls."
    )
    user = _user_prompt(
        pdb_id=pdb_id, variant=variant, split=split,
        model_pK=model_pK, rationale=rationale, initial_rag=initial_rag,
    )

    spend_mod.check_or_429()
    resp = await messages_create(
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[],   # no tools in fast mode
        max_tokens=1024,
        temperature=0.0,
    )

    text = _extract_text(resp["content"])
    try:
        verdict = _parse_verdict(text)
        verdict_dict = verdict.model_dump()
    except Exception as e:
        verdict_dict = {
            "scores": {
                "structural_consistency": 0.0, "physical_consistency": 0.0,
                "literature_consistency": 0.0, "chemical_plausibility": 0.0,
            },
            "verified_claims": [], "contradicted_claims": [],
            "missing_claims": [], "recommendation": "review", "citations": [],
            "independence_caveats": [f"fast judge malformed: {e}"],
            "_degraded": True,
        }

    trace: list[TraceStep] = [
        {"step": -1, "tool": "lookup_split", "input": {"pdb_id": pdb_id},
         "result": split, "latency_ms": 0},
        {"step": -1, "tool": "rag_query",
         "input": {"query": f"binding mode of {pdb_id}", "pdb_id": pdb_id, "top_k": 8},
         "result": _strip_rag_for_trace(initial_rag), "latency_ms": 0},
    ]
    return _attach_envelope(
        verdict_dict, trace,
        resp["usage"]["input_tokens"], resp["usage"]["output_tokens"],
        resp["usage"]["cache_read_input_tokens"],
    )
