# Supplementary Material for "Structural basis for FcµR recognition of IgM"

This directory contains the supplementary information, source data, analysis scripts, manuscript figures, and FoldX output files accompanying the manuscript.

## Folder structure

### Supplementary_Information.pdf
Main supplementary document containing:
- Supplementary Table S1: Residue contacts per FcμR-D1 chain (≤ 5 Å)
- Supplementary Table S2: RMSD values of alanine mutants
- Supplementary Table S3: Software tools and versions

### Source_Data/
Raw data underlying the figures and analyses.
- `source_data_fig2-4_all_contacts.csv` – contact proportions for all residue pairs across seven complexes (Fig. 2–4)
- `source_data_fig3_three_groups.csv` – group‑averaged contact proportions for dimer/pentamer/sIgM (Fig. 3)
- `source_data_fig4_stoichiometry_full.csv` – full contact matrix under 1, 4, or 8 FcμR chains (Fig. 4)
- `source_data_dimer_vs_pentamer_diff.csv` – differential contact analysis (dimer vs pentamer)
- `source_data_sIgM_vs_pentamer_diff.csv` – differential contact analysis (sIgM vs pentamer)
- `source_data_fig7_mafft_alignment.fasta` – multiple sequence alignment of nine mammalian FcµR‑D1 orthologs (Fig. 7)

### Scripts/
Code to reproduce the main‑text figures and the associated README.
- `fig1a_rmsd_heatmap.py` – pairwise Cα RMSD heatmap (Fig. 1A)
- `fig2-4_contact_analysis.py` – contact proportion analysis and figures (Fig. 2–4), also writes Source_Data CSV files
- `fig8_workflow.mmd` – Mermaid source for the computational workflow (Fig. 8)
- `README.md` – detailed script usage instructions

### Figures/
High‑resolution manuscript figures (PDF, 1200 dpi).
- `Fig1.pdf` – Pairwise Cα RMSD heatmap and structural superposition
- `Fig2.pdf` – High‑frequency contact heatmap and oligomer selectivity bar chart
- `Fig3.pdf` – Three‑group contact proportion bar chart
- `Fig4.pdf` – Stoichiometry‑dependent contact heatmap
- `Fig5.pdf` – Atomic visualization of the binding interface
- `Fig6.pdf` – Evolutionary conservation analysis
- `Fig7.pdf` – Multiple ortholog alignment and phylogenetic tree
- `Fig8.pdf` – Integrated computational workflow

### Foldx_config/
FoldX alanine scanning raw outputs and execution script.
- `run_alanine_scan.bat` – Batch script to reproduce the FoldX AlaScan analysis
- `foldx_alanine_scan_7yte.fxout`
- `foldx_alanine_scan_7ytc.fxout`
- `foldx_alanine_scan_7ytd.fxout`
- `foldx_alanine_scan_7ysg.fxout`
- `foldx_alanine_scan_8bpe.fxout`
- `foldx_alanine_scan_8bpf.fxout`
- `foldx_alanine_scan_8bpg.fxout`

All `.fxout` files contain per‑mutation ΔΔG values; the batch script calls FoldX with the exact parameters used in the study.

## Usage notes
- Python scripts require Python 3.12 with standard scientific packages (see `Scripts/README.md`).
- FoldX analysis requires a license for FoldX 5.0; the provided `.fxout` files are the original program output and serve as the primary data source.
- PDB structures are available from the RCSB Protein Data Bank (accession codes: 7YTE, 7YTC, 7YTD, 7YSG, 8BPE, 8BPF, 8BPG).