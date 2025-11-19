# Inverse CPU Microarchitectural Design Space Exploration

This repository provides the dataset, code, and trained model for the **inverse design** framework proposed in our submission. The goal of this work is to directly generate feasible CPU microarchitectural configurations from target PPA (performance, power, area) requirements, enabling rapid design-space exploration without large simulation sweeps.

The core contribution is a hybrid TabTransformer model that jointly predicts 18 mixed-type microarchitectural parameters and produces valid configurations at millisecond scale.

---

## Repository Structure
```
inverse_dse/
├── README.md
├── model/
│   ├── hybrid_tabtransformer_best.pt
│   └── hybrid_preprocessors.pkl
├── notebook/
│   └── inverse_dse_pipeline.ipynb
├── dataset/
│   └── gem5_mcpat_stats_results_cleaned_pruned.csv
└── dataset_creator/
    ├── para_run_sweep.py
    └── randomize_cols.py

```
