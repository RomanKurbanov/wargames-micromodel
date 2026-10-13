---
name: machine-learning-practical
description: Practical machine learning guidance for engineering projects. Use when designing, implementing, debugging, evaluating, or integrating ML systems in real products. Focuses on problem framing, data quality, feature/label design, baselines, evaluation, deployment boundaries, and common failure modes. Useful across domains including games, business systems, automation, ranking, scoring, forecasting, and classification.
compatibility: Designed for code agents working in mixed product/ML repositories. Applicable across languages and stacks; especially useful when Python tooling and non-Python runtime systems must interoperate.
metadata:
  author: openai
  version: "1.0"
---

# Machine Learning Practical

Use this skill when working on an ML problem in a real software project.

This skill is for practical ML engineering, not ML research. It helps an agent make good decisions about:
- whether ML is appropriate at all
- how to frame the problem
- how to design labels, features, and data
- how to choose a simple first model
- how to evaluate correctly
- how to integrate the result into a product safely

## Core mindset

Treat ML as a tool for approximating a useful mapping under uncertainty.

Prefer:
- clear problem framing
- simple baselines
- good data
- stable interfaces
- product-valid evaluation
- debuggable systems

Do not assume a more complex model is a better solution.

## First question: should this be ML?

Before proposing ML, check whether the problem is better solved by:
- rules
- search
- heuristics
- deterministic formulas
- thresholds
- retrieval
- ranking with hand-designed scores
- product logic

ML is most useful when:
- the mapping is too fuzzy or high-dimensional for simple rules
- examples are available or can be generated
- the desired output is stable enough to learn
- evaluation can be defined
- some error is acceptable

If the correct behavior is fully specified and deterministic, start with code, not ML.

## Default workflow

1. Define the task in product terms
2. Define the input and output schema
3. Define what “good” means
4. Build a non-ML baseline
5. Build or inspect the dataset
6. Train the simplest useful model
7. Evaluate offline
8. Evaluate in the real product context
9. Add observability and safeguards
10. Only then consider more model complexity

## Problem framing rules

Always make the task explicit:
- input type
- output type
- decision frequency
- latency constraints
- acceptable error
- business or gameplay consequence of mistakes

Translate vague goals like “be smart” into one of:
- classification
- regression
- ranking
- retrieval
- anomaly detection
- forecasting
- structured prediction
- policy modulation

If a problem can be reduced to a smaller, cleaner prediction target, prefer that reduction.

## Data rules

Data quality matters more than model cleverness.

Check:
- where labels come from
- whether labels are consistent
- whether train and runtime inputs match
- whether the dataset covers real cases, not just easy ones
- whether there is leakage
- whether class imbalance or target skew exists
- whether edge cases are represented

If labels are noisy, improve labeling before increasing model complexity.

## Feature rules

Prefer features that encode useful signal over raw dumps.

Good features are:
- stable
- available at runtime
- correctly scaled
- semantically meaningful
- aligned with the prediction task

Be suspicious of:
- using future information by accident
- features unavailable in production
- IDs that let the model memorize
- brittle text parsing when structured fields exist
- duplicate or highly redundant inputs without reason

If domain-derived features are easy to compute, usually include them.

## Label design rules

Poor labels create poor models.

Check:
- is the target actually what the product needs?
- is the target measurable?
- is it too subjective?
- is it too noisy?
- is the target delayed or indirect?
- are multiple concepts being collapsed into one label?

When possible, prefer:
- well-defined targets
- decomposed outputs
- structured outputs over vague scalar labels
- high-level controls over low-level internals

## Model selection rules

Start simple.

Default order:
1. rules baseline
2. trivial statistical baseline
3. linear model or tree baseline
4. small neural model if needed
5. only then more complex approaches

For tabular data, strong simple choices often include:
- linear/logistic regression
- gradient boosting
- random forests
- small MLPs

For fixed numeric inputs, do not default to LLMs.

Use a model class that matches the data:
- tabular model for structured numeric/categorical data
- encoder model for text classification/regression
- sequence models only when temporal dependence truly matters
- retrieval when the task is knowledge lookup, not prediction

## Evaluation rules

Never trust training loss alone.

Evaluate at two levels:
- model-level metrics
- product-level behavior

Examples of model-level metrics:
- accuracy
- precision/recall/F1
- ROC-AUC / PR-AUC
- MAE / MSE
- rank correlation
- calibration

Examples of product-level checks:
- does the system behave better?
- are failure cases acceptable?
- is the output stable?
- is latency acceptable?
- can users understand the result?
- does it beat the baseline in the real workflow?

Always compare against a simple baseline.

## Integration rules

Treat preprocessing as part of the model.

If a runtime system consumes the model:
- define exact feature order
- export normalization/scaling config
- export categorical encodings
- define output order and semantics
- keep the interface stable and explicit

The most common integration bugs come from mismatched preprocessing, not the model weights.

## Common pitfalls

### 1. Solving the wrong problem
A model performs well on the chosen target but does not improve the actual product.

Fix by redefining the target and evaluation around real use.

### 2. Leakage
The model sees information during training that would not be available at inference time.

Fix by auditing every feature and split strategy.

### 3. Good offline metrics, bad real behavior
The test set does not reflect production conditions.

Fix by evaluating on realistic scenarios and adding product-level tests.

### 4. Noisy labels
The target is inconsistent or subjective.

Fix the labeling process first.

### 5. Overcomplicated architecture
The system becomes hard to debug, deploy, or explain.

Prefer the simplest model that meets the need.

### 6. No baseline
There is no proof the model is better than rules or a trivial predictor.

Always build a baseline first.

### 7. Ignoring calibration
Outputs look like probabilities but are not trustworthy.

Check calibration if decisions depend on confidence.

### 8. No observability
You cannot tell why predictions are wrong in production.

Add logging, sampled traces, feature dumps, and prediction inspection.

### 9. Distribution shift
Training data does not match real runtime data.

Monitor runtime input distributions and compare to training.

### 10. Trying to make one model do everything
A broad, vague target often performs worse than several narrow tasks.

Prefer decomposition when possible.

## Practical debugging order

When the system is not working, inspect in this order:
1. target definition
2. dataset correctness
3. split correctness / leakage
4. feature extraction
5. preprocessing consistency
6. baseline performance
7. model predictions on hand-picked examples
8. runtime integration
9. only then model architecture changes

## Product-facing advice

Prefer systems that are:
- inspectable
- stable
- easy to tune
- easy to compare against baselines
- safe under failure

If ML controls something dynamic, add:
- smoothing
- clamps
- fallbacks
- hard constraints
- confidence-aware handling

## What good ML contributions look like

Useful contributions include:
- a cleaner task definition
- better dataset schemas
- better feature contracts
- a stronger baseline
- realistic evaluation harnesses
- error analysis reports
- runtime telemetry
- model-card-like documentation for the team
- safer integration boundaries

## Anti-patterns

Avoid pushing the project toward:
- model complexity without evidence
- vague targets like “smartness”
- hidden preprocessing
- benchmark chasing without product value
- evaluation on unrealistic data
- end-to-end learning where structured decomposition is clearer
- runtime dependence on fragile tooling when a portable artifact would do

## Response style for this project

When helping on an ML-related repository:
- be concrete
- ask what the output is supposed to do in the product
- propose the simplest testable path
- suggest baselines before advanced models
- point out leakage and data-contract risks
- keep architecture grounded in deployment reality

For deeper guidance, see:
- references/REFERENCE.md