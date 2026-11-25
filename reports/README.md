# Reports and results

| Stage | Report |
| --- | --- |
| Dataset provenance, quality and splits | [data_audit.md](data_audit.md) |
| Dataset validation and cleaning | [cleaning.md](cleaning.md) |
| Initial 1,000-row Decision Tree | [baseline_evaluation.md](baseline_evaluation.md) |
| Full dataset, original manifest protocol | [evaluation/](evaluation/) · [Report](evaluation/evaluation_report.md) |
| Full dataset, Anaïs protocol | [evaluation_anais/](evaluation_anais/) · [Report](evaluation_anais/evaluation_report.md) |
| Soukaina's evaluation synthesis | [EVALUATION_RESULTS_SOUKAINA.md](EVALUATION_RESULTS_SOUKAINA.md) |

The evaluation directories retain their own saved models, metrics, predictions and error tables. Do not mix the two protocols: their test partitions and thresholds differ. [audit_metrics.json](audit_metrics.json) contains audit metrics.
