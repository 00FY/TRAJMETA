# Transfer study protocol

This is a locally prespecified, small CPU study, not an externally registered confirmatory trial. The source and all decisions are retained. The broader benchmark proposed in the original synopsis remains outside the evidence supplied by this study.

## Data and leakage controls

The study audits 40 UCI candidates using individual licence evidence, documented feature/target roles, duplicate checks, class counts, split feasibility, and bounds of 5,000 rows and 60 features. It admits 16 datasets: 11 meta-training, two meta-validation, and three meta-test. Original held-out assignments and processed checksums are preserved. Wisconsin variants are grouped within training. Exact-feature duplicates with conflicting targets are rejected because they need a grouped protocol. This does not establish exhaustive semantic duplicate detection across UCI and OpenML.

`data/audits/study-v1/candidate_decisions.csv` records every admission or exclusion. `audit.json` contains creator attribution, DOI, licence, source hashes, dropped columns and class counts. Temporary access failures are pending, not scientific exclusions. The successful retry resolved every pending source. No scores were used to admit new tasks.

The outer 25% row split uses seed 2026. Search sees only outer-training rows. Two fixed stratified inner folds are shared by all methods. All fitted transforms stay inside each fold. Outer labels are used only after final searches. The final test tasks are Iris, German Credit, and Sonar; validation uses Glass and Banknote.

## Representation and transfer

Each trajectory has four pipeline decisions. Tokens contain the 37-value pre-action observation, 12-way action, a masked reward zero, and step fraction. One 32-hidden-unit LSTM handles initial state-only queries, prefixes, and full trajectories. A scalar head predicts within-dataset quality rank. Seed 47 is held out from encoder optimization; its trajectories remain training-task data and may enter the frozen retrieval bank. Whole-run holdout is not unseen-task validation.

The diagnostic gate requires quality AUC above 0.55, positive mean within-task Spearman correlation, noncollapsed embeddings, and decreasing training loss. Prefix AUC, simple action kNN AUC, prediction error and cosine concentration are also reported. Passing this gate establishes a limited sanity check, not transfer benefit.

Retrieval first shortlists three tasks by standardized training-only metadata, then ranks the top-quality source trajectories by cosine similarity. Gate thresholds and shaping strengths are chosen using validation inner anytime scores. The bank and encoder remain fixed during search.

Offline behavior cloning runs 20 updates on target-state templates for retrieved source configurations. It consumes no target evaluations. Every target search then receives 32 evaluations with repeats and failures charged. Shaped PPO uses the same initialization as retrieval-only PPO and adds `lambda * (gamma * phi_next - phi_current)`, with bounded potential and zero terminal potential. Starting potential is zero. Discounted shaping differences telescope over an episode; this is a learning aid, not an extra terminal objective. No policy-invariance theorem is claimed for the partially observed PPO implementation.

## Comparisons and selection

The nine methods are Random Search, Optuna, plain PPO, metadata warm start, learned retrieval warm start, learned retrieval plus shaping, source-best final-configuration warm start, nonlearned action retrieval, and forced-cold retrieval. The cold method must reproduce plain PPO's exact configurations and scores for every seed and task.

Seeds are 11, 29 and 47. Four threshold/strength settings are tried on validation only. The target score is fixed to 0.9 times the median best validation Random Search inner score; it is not a per-test-task optimum or N90 oracle. Unreached targets remain null, with budget+1 used only for an explicitly capped descriptive statistic. The final protocol locks source, configuration, audit, bank, encoder and selected parameters before final evaluation.

PPO's modest network and 128 transitions per run make this a short-budget comparison. All methods have equal evaluation counts; offline preparation and controller time differ. Search time includes controller work; bank generation and encoder training are separate overhead. The tiny test pool, score target choice, action aliases, source quality, source-task identity and short learning horizon limit interpretation.

## Reproduction

From the repository with its pinned Python environment:

```powershell
.\.venv\Scripts\python.exe -m trajmeta.benchmark restore --output data/audits/study-v1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m trajmeta.study --phase prepare
.\.venv\Scripts\python.exe -m trajmeta.study --phase validation
.\.venv\Scripts\python.exe -m trajmeta.study --phase all
.\.venv\Scripts\python.exe scripts/report_study.py
```

Prepare runs training bank generation and encoder sanity checks. Validation freezes choices. All resumes matching completed units and performs final evaluation under the lock. An interrupted unit is deliberately not silently overwritten. A source/configuration change invalidates matching run signatures; use a new study version for changed research conditions.

The original 864-evaluation pilot is preserved. Its 155 warning-bearing trials are documented in `reports/study-v1/warning_audit.json`. The updated evaluator handles undefined constant-feature ANOVA scores explicitly and raises logistic iteration limits to 5,000. Remaining warnings are retained, not hidden.

The final report uses dataset-level paired differences, averaging seeds within tasks. With three test tasks, exact two-sided sign-flip p-values cannot fall below 0.25. Bootstrap intervals are descriptive; no broad or statistically established 30% improvement is claimed.
