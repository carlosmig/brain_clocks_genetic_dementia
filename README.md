# 🧠 **Brain clocks in Down syndrome & Alzheimer’s**  
**_Submitted to_ Nature Medicine (peer review) — 2025**  

**Authors:** Carlos Coronel-Oliveros, Eimear McGlinchey, …, Juan Fortea, & Agustin Ibañez  

---

## ✨ **Project overview**

- **Brain-age clocks** trained in healthy controls (HCs) across **MRI–GMV**, **fMRI–FC**, and **EEG/MEG–FC**, with accurate predictions.  
- **Stage-dependent acceleration:** HCs≈nCar < aCar < sAD < aDS < pDS < dDS; **largest effects in DS**.  
- **Early detection:** M/EEG reveals **preclinical** acceleration in PSEN1 asymptomatic carriers (aCar).  
- **Clinico-pathological links:** Higher BAGs ↔ **worse cognition (TRS)**, **greater amyloid (centiloids)**, **higher plasma p-tau217** and **NfL**.  
- **Mechanisms & targets:** Biophysical modeling suggests hyper→hypo-excitability trajectories; **cholinergic** perturbations reduce BAGs, **GABAergic** increase BAGs; **region-specific** targets identified.

---

## 🗺️ **Parcellation & data spaces**

All modalities are mapped to the **AAL** parcellation:

- **DTI–SC** (structural connectivity)  
- **MEG–FC** and **EEG–FC** (functional connectivity)  
- **fMRI–FC**  
- **MRI–GMV** (gray matter volume)  
- **PET** receptor/transporter maps  
- **Gene-expression** maps

**AAL page:** https://www.gin.cnrs.fr/en/tools/aal/  
**Citation:** Tzourio-Mazoyer, N., *et al.* (2002). Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain. *NeuroImage*, 15(1), 273–289.

---

## 📁 **Repository structure**

> Everything here is for **plotting BAG differences** between groups and **associations** with biological/clinical biomarkers.  
> The clock folders contain **training** (HC-only), **prediction**, and **plotting** code for each modality.

```
BAGs_and_associations/
├── Zdata_FMRI/           # preprocessed/aux data for fMRI plots & associations
├── Zdata_MEG&EEG/        # preprocessed/aux data for EEG/MEG plots & associations
├── Zdata_MRI/            # preprocessed/aux data for MRI plots & associations
└── Plotting.py           # unified plotting & stats (Welch ANOVAs, FDR-BH t-tests, correlations)

Clock_EEG/
├── EEG_data.csv.rar      # EEG FC table (RAR archive; extract before use)
├── MEG_data.csv.rar      # MEG FC table (RAR archive; extract before use)
└── EEG_Model.py          # EEG/MEG clock training (HCs), prediction & plots

Clock_MRI/
├── MRI_data.csv.rar      # MRI GMV table (RAR archive; extract before use)
└── MRI_Model.py          # MRI clock training (HCs), prediction & plots

Clock_fMRI/
├── fMRI_data.csv.rar     # fMRI FC table (RAR archive; extract before use)
└── fMRI_Model.py         # fMRI clock training (HCs), prediction & plots
```

> **Note:** Please **extract all `.rar` archives** to obtain the `.csv` tables before running scripts.

---

## ⚙️ **Installation**

We recommend **Python ≥ 3.9**.

```bash
pip install pandas matplotlib statsmodels seaborn numpy scipy scikit-learn numba
```

- **pandas** — data frames → https://pandas.pydata.org/  
- **matplotlib** — plotting → https://matplotlib.org/  
- **statsmodels** — GLM/ANOVA/stats → https://www.statsmodels.org/  
- **seaborn** — statistical graphics → https://seaborn.pydata.org/  
- **numpy** — arrays & math → https://numpy.org/  
- **scipy** — scientific computing & stats → https://scipy.org/  
- **scikit-learn** — ML (SVR/SVM, CV, pipelines) → https://scikit-learn.org/  
- **numba** — required for whole-brain modeling → https://numba.pydata.org/

*Optional (for some figure/modeling utilities):*  
```bash
pip install nilearn nibabel abagen
```
- **nilearn**: neuroimaging visualization → https://nilearn.github.io/  
- **nibabel**: NIfTI I/O → https://nipy.org/nibabel/  
- **abagen**: transcriptomics → https://abagen.readthedocs.io/en/stable/

---

## ▶️ **How to run**

1. **Extract data**: Unpack all `*.csv.rar` in each `Clock_*` folder.  
2. **Install** the packages above.  
3. **Train clocks & plot performance**  
   - MRI GMV:  
     ```bash
     python Clock_MRI/MRI_Model.py
     ```
   - fMRI FC:  
     ```bash
     python Clock_fMRI/fMRI_Model.py
     ```
   - EEG/MEG FC:  
     ```bash
     python Clock_EEG/EEG_Model.py
     ```
   These scripts train on **HCs**, predict across groups, compute **BAGs**, and display performance plots.
4. **Group comparisons & associations**  
   ```bash
   python BAGs_and_associations/Plotting.py
   ```
   - Welch’s **ANOVAs** and **pairwise Welch t-tests** with **FDR-BH** correction  
   - **Correlations**: **p-tau217**, **NfL**, **TRS**, **PET/centiloids**, **longitudinal BAGs** 

---

## 🧪 **Methods notes**

- **BAG** = *predicted age – chronological age*.  
- **Models**: SVM/SVR variants trained in **HCs** per modality; evaluation via cross-validation.  
- **Comparisons**: Heteroscedastic **Welch’s ANOVA** and **pairwise Welch t-tests**; **FDR-BH** multiple-comparisons control.  

---

## 🧩 **Modeling resources**

### **Structural connectivity (SC) matrices — ready for modeling**
- OSF: https://osf.io/yw5vf/  
- **Reference:**  
  Škoch, A., Rehák Bučková, B., Mareš, J., *et al.* (2022). Human brain structural connectivity matrices – ready for modelling. *Scientific Data*, 9(1), 486.

### **Whole-brain modeling (simulated activity)**  
- **EEG-Dementias** (EEG-like): https://github.com/carlosmig/EEG-Dementias  
  - Coronel-Oliveros, C., Gómez, R. G., Ranasinghe, K., *et al.* (2024). *Alzheimer’s & Dementia*, 20(5), 3228–3250.  
  - Coronel-Oliveros, C., Lehue, F., Herzog, R., *et al.* (2025). *bioRxiv*.  
- **Homo_DMF** (fMRI-like): https://github.com/carlosmig/Homo_DMF  
  - Mindlin, I., Coronel-Oliveros, C., Sitt, J. D., *et al.* (in preparation).

### **PET receptor/transporter maps**
- Repo: https://github.com/netneurolab/hansen_receptors  
- **Reference:** Hansen, J. Y., Shafiei, G., Markello, R. D., *et al.* (2022). *Nature Neuroscience*, 25(11), 1569–1581.

### **Gene expression (transcriptomics)**
- **abagen** docs: https://abagen.readthedocs.io/en/stable/  
- **References:**  
  - Markello, R. D., Arnatkeviciute, A., Poline, J.-B., *et al.* (2021). Standardizing workflows in imaging transcriptomics with the abagen toolbox. *eLife*, 10, e72129.  
  - Allen, T. E., Herrgård, M. J., Liu, M., *et al.* (2003). Genome-scale analysis of the uses of the Escherichia coli genome: model-driven analysis of heterogeneous data sets. *Journal of Bacteriology*, 185(21), 6392–6399.

---

## 🤝 **Support & issues**

If you have questions or encounter problems, please open an **Issue** in this repository.

## Dependencies


### 🧠 Brain plotting & transcriptomics (Spin test)

For cortical surface visualization and spatially-constrained permutation (spin) tests used in the transcriptomic analyses:

- **nilearn** — statistical learning for neuroimaging in Python — *install:* `pip install nilearn` — docs: https://nilearn.github.io/
- **netneurotools** — network neuroscience utilities — *install:* `pip install netneurotools` — docs: https://netneurotools.readthedocs.io/en/latest/
- **brainsmash** — surrogate maps preserving spatial autocorrelation (spin tests / variogram-matched nulls) — *install:* `pip install brainsmash` — docs: https://brainsmash.readthedocs.io/en/latest/
- **neuromaps** — common neuroimaging coordinate systems & map transforms — *install:* `pip install neuromaps` — PyPI: https://pypi.org/project/neuromaps/
- **surfplot** — high-quality cortical surface plotting — *install:* `pip install surfplot` — PyPI: https://pypi.org/project/surfplot/
- **nibabel** — read/write neuroimaging files (NIfTI, GIfTI, etc.) — *install:* `pip install nibabel` — PyPI: https://pypi.org/project/nibabel/


## 📚 Citation

If you use this repository, please cite:

**Coronel‑Oliveros, C., McGlinchey, E., ... , Fortea, J., & Ibáñez, A. (2025). _Brain clocks chart genetic risk, staging, mechanisms, and multimodal phenotypes across Down syndrome and Alzheimer’s disease_. Manuscript submitted for peer review at *Nature Medicine*.**

Also cite AAL and any external resource you reuse (SC OSF, PET maps, abagen, modeling repos, etc.).

```

## 📬 Contact

For questions or collaboration inquiries, please contact: **Carlos Coronel‑Oliveros** 📧 <carlos.coronel@gbhi.org>

