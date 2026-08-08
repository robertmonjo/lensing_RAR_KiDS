# Script naming — scheme and run order

Self-describing naming: each figure generator is named by its figure position in the paper, and the script name, its output file and the `\includegraphics` name in the paper source all share the same base name.

## Naming map

| Script | Type | Purpose | Output file (= `\includegraphics` name) |
|---|---|---|---|
| `run_all.py` | entry point | orchestrates the whole pipeline | everything below |
| `hmg_model.py` | library (imported, not run) | constants, HMG/MOND/CDM model, KiDS data, `./data` loaders | — |
| `make_fig1_kids_rar.py` | figure | Fig. 1 (body) — KiDS-1000 RAR: 4 mass bins + morphology | `outputs/fig1_kids_rar.{png,pdf}` |
| `make_figA1_regimes.py` | figure | Fig. A.1 (appendix) — two competing terms in ξ²ₛ | `outputs/figA1_regimes.{png,pdf}` |
| `make_figA2_landscape.py` | figure | Fig. A.2 (appendix) — regime map (a) + χ²ᵥ(s) landscape (b) | `outputs/figA2_landscape.{png,pdf}` |
| `make_reference_systems.py` | data | external-system gold stars for Fig. A.2 | `outputs/reference_systems.csv` (+ per-object CSVs) |
| `make_tables.py` | tables | model-comparison + MICE tables | `outputs/tab_model_comparison.csv`, `tab_mice_comparison.csv` |
| `xcop_rnei_nsig.py` | data | X-COP n_sigma from the r_nei model | `outputs/xcop_rnei_nsig.txt` |
| `make_tab_regimes.py` | table | Table A.1 (regimes) | `outputs/tab_regimes.csv`, `tab_regimes_body.tex` |

Figure position ↔ paper label. **Labels are kept as-is**; only the figure *file names* follow the numbering scheme (1 body; A.1, A.2 appendix):

| paper position | paper label | figure file (script, output, `\includegraphics`) |
|---|---|---|
| Fig. 1 (body)       | `fig:kids_rar`       | `fig1_kids_rar` |
| Fig. A.1 (appendix) | `fig:hmg_regimes`    | `figA1_regimes` |
| Fig. A.2 (appendix) | `fig:combined_fig45` | `figA2_landscape` |

## Run order

One command:

```
python run_all.py
```

`hmg_model.py` is a library (imported, never run alone); the figure/table generators import it for the shared model and KiDS data. Dependencies handled by `run_all.py`: `make_reference_systems` → `make_figA2_landscape`; and `make_tables` + `make_reference_systems` + `xcop_rnei_nsig` → `make_tab_regimes`.
