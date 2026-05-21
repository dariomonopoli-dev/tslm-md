"""MDCoTQADataset — yields the OpenTSLM 5-key dict for binding-affinity prediction.

Stub: implement during hour 1-2.

Subclass `opentslm.time_series_datasets.QADataset` and yield per sample:
    {
      "time_series":      Tensor[6, 30],
      "time_series_text": ["MD trajectory features per frame"],
      "pre_prompt":       prompts.build_prompts()[0],
      "post_prompt":      prompts.build_prompts()[1],
      "answer":           "Answer: <x> kcal/mol. Confidence: <high|medium|low>.",
    }

Data sources:
  - data/featurized.h5     (precomputed by scripts/preprocess_features.py)
  - data/pdbbind_index/    (PDB id -> -logKd/Ki)
  - data/splits/{train,val,test}.txt
"""
# TODO(hour 1-2): implement
