# Supplementary Material for "Structural basis for FcµR recognition of IgM"

This directory contains supplementary information, source data, analysis scripts,
manuscript figures, and FoldX output files accompanying the manuscript.

================================================================================
FOLDER STRUCTURE
================================================================================

supplementary_information.pdf
    Main supplementary document containing the table of contents and
    descriptions of all supplementary files.

supplementary_table_s1.xlsx
    Supplementary Table S1: Residue contacts per FcμR-D1 chain (≤ 5 Å)

supplementary_table_s2.xlsx
    Supplementary Table S2: RMSD values of alanine mutants

source_data/
    Raw data underlying the figures and analyses.

    - source_data_fig2-4_all_contacts.csv
      Contact proportions for all residue pairs across seven complexes (Fig. 2–4)

    - source_data_fig3_three_groups.csv
      Group-averaged contact proportions for dimer, pentamer, and sIgM (Fig. 3)

    - source_data_fig4a_stoichiometry_full.csv
      Full contact matrix under 1, 4, and 8 FcμR chains (Fig. 4A)

    - source_data_fig4b_pentamer_j_chain_stoichiometry.csv
      Filtered pentamer J-chain stoichiometry data for heatmap (Fig. 4B)

    - source_data_dimer_vs_pentamer_diff.csv
      Differential contact analysis: dimer vs pentamer

    - source_data_sIgM_vs_pentamer_diff.csv
      Differential contact analysis: sIgM vs pentamer

    - source_data_fig7_mafft_alignment.fasta
      Multiple sequence alignment of nine mammalian FcµR-D1 orthologs (Fig. 7)

scripts/
    Code to reproduce the main-text figures and run the alanine scanning
    preprocessing.

    - renumber_chains.py
      PDB chain renumbering using Biopython, required before FoldX AlaScan

    - fig1a_rmsd_heatmap.py
      Pairwise C-alpha RMSD heatmap (Fig. 1A)

    - fig2-4_contact_analysis.py
      Contact proportion analysis and figures (Fig. 2–4), also writes source_data
      CSV files

    - fig8_workflow.mermaid
      Mermaid source for the computational workflow (Fig. 8)

    - README.md
      Detailed script usage instructions

figures/
    High-resolution manuscript figures (jpg, 1200 dpi).

    - fig1a.jpg
      Pairwise C-alpha RMSD heatmap and structural superposition (Fig. 1)

    - fig1b.jpg
    Structural superposition aligned on the FcµR-D1 chain (Fig. 1B)

    - fig2a.jpg
      High-frequency contact heatmap (Fig. 2A)

    - fig2b.jpg
      Oligomer selectivity stacked bar chart (Fig. 2B)

    - fig3.jpg
      Three-group contact proportion bar chart (Fig. 3)

    - fig4a.jpg
      Pentameric IgM stoichiometry-dependent contact heatmap (Fig. 4A)

    - fig4b.jpg
      Pentameric J-chain stoichiometry heatmap (Fig. 4B)

    - fig5a.jpg
      ConSurf conservation grade of FcµR (Fig. 5A)

    - fig5b.jpg
      Domain organization of FcµR (Fig. 5B)

    - fig5c.jpg
      Conservation scores mapped on FcµR-D1 (Fig. 5C)

    - fig6a.jpg
      Multiple ortholog alignment across nine mammalian species (Fig. 6A)

    - fig6b.jpg
      Hierarchical MAFFT CLUSTAL phylogenetic tree (Fig. 6B)

    - fig7a.jpg
      Overview of the FcμR-D1–Cμ4 crystal complex (Fig. 7A)

    - fig7b.jpg
      Arg45-Glu468 ion pair and Phe67-Glu468 hydrophobic packing (Fig. 7B)

    - fig7c.jpg
      Lys69-Glu526 ion pair and Asp111-Arg467 ion pair (Fig. 7C)

    - fig7d.jpg
      Thr60-Arg467 and Ser63-Glu468 hydrogen bonds (Fig. 7D)

    - fig7e.jpg
      Thr110-Arg467 hydrogen bond (Fig. 7E)

    - fig7f.jpg
      FcµR-D1 high-contact residue pairs with the C-terminal region of the J-chain (Fig. 7F)

    - fig7g.jpg
      FcµR-D1 molecules (R1′–R4′) bound to sIgM, opposite the SC (Fig. 7G)

    - fig8.jpg
      Integrated computational workflow (Fig. 8)

foldx_config/
    FoldX alanine scanning raw outputs and execution commands.

    - foldx_input_config.txt
      Run commands and parameters for all seven complexes

    - foldx_alanine_scanning_7yte.fxout
    - foldx_alanine_scanning_7ytc.fxout
    - foldx_alanine_scanning_7ytd.fxout
    - foldx_alanine_scanning_7ysg.fxout
    - foldx_alanine_scanning_8bpe.fxout
    - foldx_alanine_scanning_8bpf.fxout
    - foldx_alanine_scanning_8bpg.fxout

    All .fxout files contain per-mutation ΔΔG values.

================================================================================
USAGE NOTES
================================================================================

- Python scripts require Python 3.12 with standard scientific packages (NumPy,
  Pandas, Matplotlib, Seaborn, Biopython, MDAnalysis). See scripts/README.md
  for details.

- FoldX analysis requires a license for FoldX 5.0. The provided .fxout files
  are the original program output and serve as the primary data source.

- PDB structures are available from the RCSB Protein Data Bank under accession
  codes: 7YTE, 7YTC, 7YTD, 7YSG, 8BPE, 8BPF, 8BPG.

- All supplementary tables (S1, S2) are provided as editable .xlsx files.

- Software and code availability information is provided in the main article.