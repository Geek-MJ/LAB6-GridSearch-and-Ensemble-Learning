# GridGuard — Audit of AI-Assisted Work

The project requires **at least four concrete problems detected in AI-generated outputs**, with **at least one genuinely detected and corrected problem per student**. Each entry must state the original flaw, how it was discovered, its consequences, the correction and verifiable evidence. Do not invent errors or claim that an issue was fixed if it was only identified.

## 1. Mahij Khantigiri GOSAI — Input Duplication Across Data Partitions

**Problem in an earlier AI-assisted implementation:** The earlier approach used a conventional row-wise stratified train/test split. Stratification preserved the target-class distribution but did not keep identical input records together.

**Detection:** The project audit reported **447 test records with complete feature vectors also present in the original training pool**. That is a duplicate-input overlap across the outer split.

**Why it matters:** A model could encounter an identical input record during training and testing, making performance estimates optimistic for unseen input configurations.

**Correction:** Complete input vectors were grouped before partitioning. Unique groups were divided into training and test partitions using stratification on the binary target, after which all 20,000 original records inherited their group's assignment. No standalone validation dataset was exported.

**Verification:** The current `train.csv` contains **16,018** records and `test.csv` contains **3,982**. Automated verification confirmed **zero identical complete feature vectors shared across the exported files**. Evidence: `src/split_data.py`, `tests/test_team_split.py`, `notebooks/01_data_audit_executed.ipynb` and `reports/team_split.md`.

**Residual limitations:** Exact-input separation does not prove that different simulation configurations are independent. Predictor availability before the simulated outcome remains unresolved.

## 2. Soukaina BAHAMI — Baseline / Preprocessing Error

**Original AI-output flaw:**  
[Describe one real AI-generated error you found in your module.]

**How it was detected and why it matters:**  
[Explain the actual failed check, incorrect result or methodological issue.]

**Correction and validation:**  
[Describe the actual fix and cite the source file, test or result that demonstrates it.]

## 3. Anais KAMTCHOUANG NONO — Advanced Model Error

**Original AI-output flaw:**  
[Describe one real AI-generated error you found in your module.]

**How it was detected and why it matters:**  
[Explain the actual failed check, incorrect result or methodological issue.]

**Correction and validation:**  
[Describe the actual fix and cite the source file, test or result that demonstrates it.]

## 4. Yutong CAO — Evaluation Error

**Original AI-output flaw:**  
[Describe one real AI-generated error you found in your module.]

**How it was detected and why it matters:**  
[Explain the actual failed check, incorrect result or methodological issue.]

**Correction and validation:**  
[Describe the actual fix and cite the source file, test or result that demonstrates it.]
