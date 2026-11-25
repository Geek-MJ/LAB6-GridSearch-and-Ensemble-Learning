# Datasets

| Location | Purpose |
| --- | --- |
| [Blackout_PowerGraph_1000.csv](Blackout_PowerGraph_1000.csv) | Original 1,000-row extract; kept here for existing code compatibility |
| [raw/](raw/) | Original 19,000-row ZIP and source metadata inside the archive |
| [processed/](processed/) | Validated 20,000-row dataset, optional 18,579 unique-input dataset, cleaning summary and unique-record manifest |
| [splits/](splits/) | Audit-generated group-aware train, validation, test and all-record manifests |

Read the [data audit](../reports/data_audit.md) and [cleaning report](../reports/cleaning.md). Do not modify original inputs. The two evaluation protocols use different partitions; see the [report index](../reports/README.md).
