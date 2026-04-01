# Machine Learning Practical Reference

This reference provides deeper guidance for transferable ML engineering work across domains.

## 1. A reusable mental model

Most practical ML systems fit this shape:

raw inputs -> feature extraction -> model -> postprocessing -> product decision

Debug each stage separately.

Do not treat the model as the whole system.

## 2. Task-framing cheat sheet

Translate vague requests into standard task types:

- “score this thing” -> regression or ranking
- “choose among categories” -> classification
- “find suspicious cases” -> anomaly detection
- “predict future value” -> forecasting
- “extract fields” -> structured prediction
- “adjust system behavior” -> policy modulation or control weights

The more tightly scoped the target, the easier the project.

## 3. Baseline strategy

Every ML project should have:
- a trivial baseline
- a simple heuristic baseline
- one simple learned baseline

Examples:
- predict the mean
- most frequent class
- threshold rules
- hand-crafted score
- linear model
- boosted trees

If the fancy model barely beats the baseline, the added complexity may not be worth it.

## 4. Label-source guidance

Common label sources:
- human annotation
- existing business decisions
- simulator outcomes
- rules engines
- delayed outcomes
- stronger model / teacher model

For each source, ask:
- what bias does this source introduce?
- what noise does it contain?
- does it reflect the desired target or only a proxy?

Teacher-generated labels can be useful, but they are still labels with bias.

## 5. Data split guidance

Choose splits that reflect deployment.

Good split strategies may include:
- time-based split for temporal systems
- entity/group split to avoid memorization
- scenario-based split for simulators
- geography/customer/team split when domains vary

Random split is often too optimistic.

## 6. Error analysis workflow

After training:
1. inspect worst errors
2. inspect representative correct predictions
3. slice by cohort or scenario type
4. inspect edge cases
5. compare baseline vs model on the same examples

This usually yields more insight than tuning hyperparameters.

## 7. Calibration guidance

If outputs are interpreted as probabilities or confidence:
- check calibration
- do not assume sigmoid outputs are calibrated
- consider calibration layers or post-hoc calibration if needed

Confidence should influence downstream decisions only if it is validated.

## 8. Distribution-shift checklist

Compare training vs runtime for:
- missing values
- feature ranges
- category frequencies
- text formats
- scenario frequencies
- target prevalence

A model can fail simply because production inputs look different.

## 9. Sequence and memory guidance

Do not jump to recurrent or sequence models immediately.

First ask whether the issue can be addressed by:
- adding lagged features
- previous outputs
- moving averages
- simple state wrappers
- smoothing or hysteresis

Use true sequence models only when temporal dependence is central.

## 10. Deployment boundary guidance

When different languages or runtimes are involved:
- define a stable schema
- export preprocessing config
- keep artifacts portable
- test one known example end-to-end across both sides

The deployment boundary is part of the design, not an afterthought.

## 11. Safe system design

When ML influences behavior in a live system:
- keep hard constraints in code
- add fallback behavior
- cap outputs when needed
- monitor drift and failures
- log enough to replay or inspect errors

Do not let the model own safety-critical invariants unless that is explicitly intended and validated.

## 12. Signs the project is healthy

Healthy ML projects usually have:
- clear target definitions
- realistic evaluation
- strong baselines
- reproducible datasets
- versioned artifacts
- observable runtime behavior
- simple explanations for why the model helps

## 13. Signs the project is drifting

Watch for:
- constant architecture switching
- no stable benchmark
- vague discussions about “intelligence”
- labels changing every week
- no one can explain what the output means
- runtime integration repeatedly breaking due to schema mismatch
- demo-driven tuning with no measurement

## 14. Good questions for any ML repository

Ask:
- what exactly is predicted?
- where do the labels come from?
- what is the simplest baseline?
- how is leakage prevented?
- what split mirrors production?
- what happens when the model is uncertain or wrong?
- how is the output consumed downstream?
- what would make us remove ML from this problem?

## 15. Contribution standard

A good ML-related change should improve at least one of:
- target clarity
- dataset quality
- evaluation quality
- baseline strength
- deployment reliability
- observability
- runtime safety
- product usefulness