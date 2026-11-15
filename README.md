# CFD-Based Analysis of Shear Stress and Oxygen Transport in a Widening Hele-Shaw Chip

CLL770 Microfluidics term project, IIT Delhi: Utkarsh Srivastav (2023CH70715), Divya (2023CH10309), Ishaan Singal (2023CH70545).

This folder holds the whole project: the reports, the raw Fluent results, all code, the processed results and every plot.

## Folder map

| Folder | What is inside |
|---|---|
| `00_Deliverables/` | Finished documents: **term paper (17 pp, submission)**, extended technical report (78 pp), CFD post-processing report (PDF + DOCX), Excel workbook of all results, project manual / Fluent set-up guide (PDF + DOCX) |
| `01_LaTeX_Source/` | `paper.tex` (term paper), `main.tex` + `chapters/` (extended report), `references.bib`, `figures/` (every figure used in either document) |
| `02_CFD_Results_Raw/` | Fluent XY exports exactly as supplied (`raw_fluent_exports/`, one folder per flow rate), Fluent contour and pathline images, residual-monitor screenshots |
| `03_Flow_PostProcessing/` | `code/` (Python analysis and figures), `results/tables/` (CSV: station comparison, pressure summary, uniformity, velocity, scaling, validation, …), `results/results_summary.json`, `Graphs_41_figures/` (41 plots in six themed folders, 300 dpi PNG + vector PDF) |
| `04_Oxygen_Model/` | `code/` (1.5-D oxygen model, closed forms, figure script), `results/oxygen_numbers.json` (every oxygen number in the paper), `figures/`, `design_study_outputs/` (parameter sweeps, Pareto designs, baseline metrics) |
| `05_Geometry_Theory_Fluent_Setup/` | `01_geometry/` (CAD scripts, STEP/DXF files, wall coordinates), `02_analytical/` (Hele-Shaw theory and 2D model T3), `03_fluent/` (Fluent journals for flow and oxygen runs, oxygen UDF `o2_transport.c`), `04_postprocessing/` (GCI/mesh-independence, shear validation and oxygen post-processing tools for future runs), `06_templates/` (mesh-study, validation and oxygen report templates) |

## Reproduce the results

Requirements: Python 3 with numpy, scipy, pandas and matplotlib; a TeX distribution with pdflatex and bibtex.

```bash
cd 03_Flow_PostProcessing/code
python analysis.py        # reads 02_CFD_Results_Raw -> 03_Flow_PostProcessing/regenerated/output
python make_figures.py    # -> 03_Flow_PostProcessing/regenerated/Graphs (41 figures)

cd ../../04_Oxygen_Model/code
python make_report_figures.py   # -> 04_Oxygen_Model/regenerated/ (figures + oxygen_numbers.json)

cd ../../01_LaTeX_Source
latexmk -pdf paper.tex    # term paper
latexmk -pdf main.tex     # extended report
```

Both pipelines were re-run from inside this folder before packaging. They reproduce the delivered tables and oxygen numbers exactly.

## Key results

- **Stokes flow.** Δp = 1.657 Pa per µL/min, and all quantities scale with Q to within 0.08 %.
- **Agreement with Eq. 2.** Wall shear agrees within ±3.3 % at S1–S4, partly a cancellation: the CFD includes side-wall drag (+9 %) and carries a −5 % offset, probably from the mesh. S5 is +46.5 % above Eq. 2 because of the outlet corner.
- **Uniform-shear zone.** ±5 % uniform over 70–91 % of the width at S1–S4. Image in the centre of S2–S4, at least 0.17 mm from the side walls.
- **Oxygen (model predictions).** With impermeable walls, cells become anoxic beyond 13.9 mm at 0.22 µL/min and 38.7 mm at 1 µL/min. The minimum flow to feed all cells is 1.55 µL/min. A 5 mm PDMS lid keeps the cells at 95 % of saturation.

## Open items

- **Mesh-independence study.** Not yet run. Use `06_templates/mesh_independence_template.csv`, the online template sheet https://claude.ai/artifact/3yAFis5PDB7KhVksaQSZdb, and `04_postprocessing/mesh_independence.py`.
- **Oxygen CFD.** Not yet run. Journals and UDF are in `03_fluent/`.
- **Inferred values.** The channel height (145 µm) and the 2 mm lead-in were inferred, not documented.
- **Full list.** Appendix A of the extended report lists all missing or assumed information.
