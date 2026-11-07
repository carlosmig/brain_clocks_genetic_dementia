# Brain clocks chart genetic risk, staging, mechanisms, and multimodal phenotypes in Down syndrome and Alzheimer’s disease

**Journal:** Nature Aging

Code and source data for the brain-age clocks and Figures 2–4.

- `clocks/` contains the EEG, fMRI, and MRI training scripts. Each script loads CSV files, trains the clock, calculates brain age gaps (BAGs), and saves a performance plot.
- `figures/` contains one source-data workbook and one plotting script per figure. The scripts reproduce the performance scatter plots in 2a, the maps and gene associations in 3c–d, and the plots supported by the supplied data in 4b–f.

## Setup

Use Python 3.12 and install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python clocks/Clock_EEG/EEG_Model.py
python clocks/Clock_fMRI/fMRI_Model.py
python clocks/Clock_MRI/MRI_Model.py

python figures/Figure2/plot_figure2.py
python figures/Figure3/plot_figure3.py
python figures/Figure4/plot_figure4.py
```

