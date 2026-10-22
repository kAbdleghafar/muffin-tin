# mtpot

Muffin-tin / EAM-like interatomic potential for bcc refractory multi-principal-element
alloys, fitted to DFT (QE + thermo_pw) energies, forces and stresses, with the DFT valence
density and ELF shaping the atom-centred functions. Target: relaxed-ion elastic constants
(C11, C12, C44) of new SQS structures at near-DFT accuracy.

See [CLAUDE.md](CLAUDE.md) for the data, conventions, known data issues and roadmap.

```bash
pip install -e ".[test]"
python -m pytest                  # sample tests skip unless data/raw/ holds the DFT files
python scripts/tpw_to_dataset.py data/raw/elast_*.out \
    -o data/processed/dataset.extxyz -s data/processed/elastic_summary.csv
```

Raw DFT files (`data/raw/`) are not in git.
