# GridGuard — Dataset provenance, quality, EDA and split audit

**Module owner:** Mahij Khantigiri GOSAI  
**Stage:** Dataset preparation and exploratory analysis  
**Data:** IEEE-24 simulation extracts provided for the course project  
**Code:** `src/audit.py` (reproducible; run with `python -m src.audit`)

## 1. Scope and provenance

The objective is to predict whether a simulated grid scenario results in **any electricity demand not being supplied**. `blackout=1` means unmet demand; it does **not** necessarily imply a national blackout. The supplied files are `Blackout_PowerGraph_1000.csv` (1,000 records) and `Blackout_PowerGraph_19000.zip` (19,000 records). The ZIP's `README.txt` attributes them to **PowerGraph**, IEEE-24, from [Figshare](https://figshare.com/articles/dataset/PowerGraph/22820534), source archive `dataset_cascades.zip` (file ID 50083479). The ZIP also contains `source_indices.json` with 19,000 source indices; the smaller CSV has no source index mapping in the supplied files. The original scenario indices cannot be compared between the two extracts because the 1,000-row file has no source-index mapping.

The provided source README notes that the original MATLAB simulation matrices were flattened to 224 input features and numerical values are stored at ten significant digits. The available documentation does not confirm the physical units or the measurement timing of each feature.

## 2. Quality audit — executed results

| Check | Observed result |
|:---|---:|
| Records combined | 20,000 |
| Source columns | 225 (224 inputs + `blackout`) |
| Target 0 / 1 | 15,968 / 4,032 |
| Positive proportion | 20.16% |
| Missing cells across features and label | 0 |
| Infinite or non-finite input cells | 0 |
| Constant input columns | 0 |
| Unique complete input patterns | 18,579 |
| Repeated records beyond first occurrence | 1,421 |
| Distinct input patterns found in **both** extracts | 152 |
| Identical input vectors with conflicting labels | 0 |

**Interpretation.** No missing-value imputation was needed. Identical input vectors can affect test results if copies are present in both training and testing. We kept these records and grouped identical inputs during splitting. Identical values do not necessarily represent the same original simulation run.

## 3. Exploratory analysis

The data contain the expected seven feature families: 24 bus net-active-power variables, 24 bus net-apparent-power variables, 24 bus voltage magnitudes, and four groups of 38 line variables (active flow, reactive flow, reactance and rating). The feature categories describe the supplied table rather than a reconstruction of the network topology.

Figures in `figures/`:
1. `01_target_by_extract.png` — observed target counts per source file.
2. `02_feature_families.png` — number of features in each family.
3. `03_duplicate_group_sizes.png` — prevalence of repeated complete input vectors.
4. `04_univariate_differences.png` — largest standardized mean differences between class labels.
5. `05_feature_correlations.png` — correlation plot for 25 highest-variance features.

The fourth figure compares the classes using the complete dataset for exploratory analysis. Feature selection for modeling should be carried out within training data and cross-validation folds.

## 4. Original group-aware split protocol (historical comparison)

The initial audit used a 70/15/15 train/validation/test split. The newer two-file split is implemented in `src/split_data.py` and documented in [team_split.md](team_split.md). The results below describe the initial experiment.

A group ID is assigned from the **complete feature vector**. Candidate 70/15/15 group partitions are generated using fixed seeds; the chosen candidate minimizes split-size and class-prevalence discrepancies. This uses target labels only to balance the partitions, not to optimize a classifier. No duplicated feature vector can cross partitions. Manifest CSVs preserve original row identifiers and, for the 19,000-row subset, provided scenario indices.

| Partition | Records | Positive cases | Positive rate | Unique feature groups |
|:---|---:|---:|---:|---:|
| Training | 13,993 | 2,820 | 20.15% | 13,005 |
| Validation | 3,010 | 607 | 20.17% | 2,787 |
| Test | 2,997 | 605 | 20.19% | 2,787 |

**Original experiment:** The manifests match records using `record_id`. Preprocessing and supervised feature selection use the training records, validation data are used for model and threshold selection, and test data are reserved for the final assessment.

## 5. Dataset limitations

1. **Feature timing:** The source documentation does not confirm when every variable is measured. Features derived after a disruption could introduce outcome leakage, so the original PowerGraph definitions remain important.
2. **Related scenarios:** Grouping identical inputs prevents exact duplicates from crossing partitions, but other records may still represent similar simulation conditions. The 1,000-row extract has no source indices for further grouping.
3. **Simulation data:** Performance on simulated IEEE-24 networks is not necessarily the same as performance on real operating networks.
4. **Sampling:** The source README describes proportional target sampling. The observed class frequencies may therefore differ from real operating conditions.
5. **Feature representation:** The flattened columns do not provide the complete network topology, so interpretation also depends on the original dataset documentation.
6. **Group identifiers:** Input groups are represented using pandas 64-bit hashes. Hash collisions are unlikely but possible; the original input files were retained.

## 6. Summary

The data contain no missing values and no conflicting labels for identical complete inputs. The main preparation issue was repeated input records, which led to the grouped splitting approach. The audit and exploratory results provide the dataset checks used before model training. Feature timing and dependencies between related simulations remain limitations.

### Reproduction

```bash
python -m pip install -r requirements.txt
python -m src.audit
python -m pytest -q
```

The notebook uses relative paths and can be run with the repository root as its working directory.
