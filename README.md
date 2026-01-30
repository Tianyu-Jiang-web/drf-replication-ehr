
# Distributional Random Forests (DRF) Replication on EHR Readmission Data

This project reproduces and extends key ideas from  
**Cevid et al. (2022), “Distributional Random Forests”**,  
with a focus on **uncertainty-aware prediction of hospital readmission**
using EHR data.

The goal is not only to predict readmission risk, but also to **quantify
predictive uncertainty** by exploiting the distributional and
weighting-based nature of DRF.

------------------------------------------------------------------------

## 1. Project Objectives

This replication focuses on the following objectives:

1.  **Replicate DRF on real-world EHR data**
    - Binary outcome: hospital readmission (0/1)
    - High-dimensional patient covariates
2.  **Demonstrate uncertainty-aware prediction**
    - Predictive probability of readmission
    - Uncertainty measures derived from DRF weights:
      - Predictive entropy
      - Effective Sample Size (ESS)
      - Credible intervals for readmission risk
3.  **Evaluate uncertainty usefulness**
    - Calibration and reliability
    - Selective prediction (coverage–risk trade-off)
    - Relationship between uncertainty and prediction error
4.  *(Optional extension)*  
    Show how DRF can support **multi-target / joint outcome modeling**
    when additional outcomes (e.g. LOS, mortality) are available.

------------------------------------------------------------------------

## 2. Data Description

### Inputs

- **X**: Patient-level EHR features, aggregated per admission,
  including:
  - Demographics
  - Comorbidities
  - Lab summaries
  - Vital signs summaries
  - Medication counts
  - Procedures
  - (Optional) text or embedding-based features
- **Y**: Binary readmission indicator (0/1)

### Optional Extension (Recommended)

To better showcase DRF’s distributional advantages, an extended outcome
vector may be used: Y_multi = (readmission, length_of_stay, mortality,
ICU_days, cost, …)

This allows exploration of: - Joint distributions - Conditional
dependence structures - Tail behavior across outcomes

------------------------------------------------------------------------

## 3. Software and Reproducibility

- Language: **R**
- Core method: **Distributional Random Forests (DRF)**
- Repository: <https://github.com/lorismichel/drf>

The project uses:

- **Git** for version control
- **renv** for dependency management

To reproduce the environment:

\`\`\`r renv::restore()
