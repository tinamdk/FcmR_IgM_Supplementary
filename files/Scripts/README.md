# Figure Generation Scripts

Scripts for reproducing main-text figures in "Structural basis for FcµR recognition of IgM".

## Requirements

- Python 3.12
- NumPy, Pandas, Matplotlib, Seaborn, Biopython, MDAnalysis
- PyMOL 3.1.6.1 (Schrodinger)
- UCSF ChimeraX 1.10

Install Python dependencies:
pip install numpy pandas matplotlib seaborn biopython mdanalysis

## Scripts

### renumber_chains.py
Pre-analysis renumbering of PDB chains using Biopython, required before running FoldX AlaScan. Ensures unique residue identifiers across identical chains.
Usage: python renumber_chains.py
Input: PDB file in source_file/ directory.
Output: Renumbered PDB file with unique residue identifiers.

### fig1a_rmsd_heatmap.py
Generates the pairwise C-alpha RMSD heatmap (Fig. 1A).
Usage: python fig1a_rmsd_heatmap.py
Input: PDB files in source_file/ directory (7YTE, 7YTC, 7YTD, 7YSG, 8BPE, 8BPF, 8BPG).
Output: fig1a_rmsd_heatmap.png and fig1a_rmsd_heatmap.pdf (1200 dpi).

### fig2-4_contact_analysis.py
Contact analysis script for Fig. 2-4. Computes residue-residue contact proportions across all seven FcµR-Fcµ complexes, including J-chain and SC contact analyses.
Usage: python fig2-4_contact_analysis.py
Input: PDB files in source_file/ directory.
Output:
  Figures (1200 dpi):
    - fig2a_contact_heatmap.png/pdf          (Fig. 2A)
    - fig2b_stacked_bar.png/pdf              (Fig. 2B)
    - fig3_three_groups_barplot.png/pdf      (Fig. 3)
    - fig4a_stoichiometry_heatmap.png/pdf    (Fig. 4A)
    - fig4b_pentamer_j_chain_heatmap.png/pdf (Fig. 4B)
  Source data CSV files written to ../source_data/:
    - source_data_fig2-4_all_contacts.csv
    - source_data_fig3_three_groups.csv
    - source_data_fig4a_stoichiometry_full.csv
    - source_data_dimer_vs_pentamer_diff.csv
    - source_data_sIgM_vs_pentamer_diff.csv
    - source_data_fig4b_pentamer_j_chain_stoichiometry.csv
  Additional contact CSV files written to the output directory:
    - fig4b_pentamer_j_chain_all_contacts.csv
    - fig4b_j_sIgM_contacts.csv

### fig8_workflow.mermaid
Mermaid source code for the integrated computational workflow (Fig. 8).
Usage: Paste contents into https://mermaid.live/ and export as PDF.

## FoldX Alanine Scanning
Raw FoldX results and execution commands for all seven complexes are in ../foldx_config/.
Each .fxout file contains per-mutation ΔΔG values.
To reproduce the analysis, download FoldX 5.0 (https://foldxsuite.crg.eu) and use the commands provided in ../foldx_config/foldx_input_config.txt.
Table 2 in the main text presents representative results from PDB 7YTC.

## PDB Files
PDB structures used in this study are publicly available from the RCSB Protein Data Bank (https://www.rcsb.org/) under accession codes 7YTE, 7YTC, 7YTD, 7YSG, 8BPE, 8BPF, and 8BPG. All scripts expect these files in a source_file/ subdirectory.