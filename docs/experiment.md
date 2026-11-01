# Hiver Support Agent - ML Experiment Log

## Purpose

This document records the experiments performed while improving the 6-class intent-classification model for the Hiver Support Agent.

It preserves:
- what was tested
- why it was tested
- dataset/taxonomy used
- results
- conclusions
- rejected directions
- next experiment

No production dataset or production training code should be changed unless later experiments provide sufficient evidence.

---

# 1. Current 6-Class Taxonomy

| 6-class intent | Original intents |
|---|---|
| CUSTOMER_SUPPORT | CUSTOMER_SUPPORT_ACCESS, CUSTOMER_SUPPORT_SERVICE_FEEDBACK |
| NETWORK_CONNECTIVITY | SLOW_INTERNET, INTERNET_OUTAGE |
| BILLING_PAYMENT | BILLING_PAYMENT |
| WIFI_ROUTER | WIFI_ROUTER |
| SWITCHING_RETENTION | SWITCHING_RETENTION |
| OTHER | OTHER |

Excluded rare classes:
- FRAUD_SECURITY
- MOBILE_PHONE_SERVICE
- REFUND
- SIM_CARD

There are 8 rare-class rows.

---

# 2. Baseline Dataset

Source:

`data/processed/merged_training_dataset_cleaned.csv`

Original dataset:
- 338 rows
- 4 columns

After removing 8 rare-class rows:
- 330 evaluation examples

Distribution:

| Class | Examples |
|---|---:|
| CUSTOMER_SUPPORT | 96 |
| SWITCHING_RETENTION | 67 |
| BILLING_PAYMENT | 58 |
| NETWORK_CONNECTIVITY | 42 |
| WIFI_ROUTER | 39 |
| OTHER | 28 |
| **Total** | **330** |

Pseudo-labeled rows are excluded when `is_pseudo_labeled` exists.

---

# 3. Baseline TF-IDF Experiment

Several word-level TF-IDF configurations were compared using 5-fold CV.

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| BASELINE | 0.6758 | 0.6635 | 0.6713 |
| WORD_BIGRAMS | 0.6697 | 0.6591 | 0.6631 |
| SUBLINEAR_BIGRAMS | 0.6848 | 0.6701 | 0.6787 |
| BIGRAMS_MIN_DF_2 | **0.6879** | **0.6797** | **0.6864** |
| BIGRAMS_MIN_DF_2_UNBALANCED | 0.6606 | 0.6061 | 0.6431 |

## Finding

The strongest configuration in this experiment was:

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
```

with balanced class weights.

Important observations:
1. Word bigrams alone did not help.
2. `sublinear_tf=True` improved performance.
3. `min_df=2` produced the best Macro F1.
4. Removing class balancing caused a substantial Macro F1 drop.

This became the working baseline.

---

# 4. 6-Class Confusion Inspection

Out-of-fold results for the best configuration:

- Accuracy: 0.6879
- Macro F1: 0.6797
- Weighted F1: 0.6864

Classification report:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| BILLING_PAYMENT | 0.80 | 0.76 | 0.78 |
| CUSTOMER_SUPPORT | 0.69 | 0.55 | 0.61 |
| NETWORK_CONNECTIVITY | 0.61 | 0.67 | 0.64 |
| OTHER | 0.44 | 0.50 | 0.47 |
| SWITCHING_RETENTION | 0.69 | 0.82 | 0.75 |
| WIFI_ROUTER | 0.82 | 0.85 | 0.84 |

Main confusion pairs:

- CUSTOMER_SUPPORT -> SWITCHING_RETENTION: 14
- CUSTOMER_SUPPORT -> OTHER: 11
- CUSTOMER_SUPPORT -> NETWORK_CONNECTIVITY: 9
- SWITCHING_RETENTION -> CUSTOMER_SUPPORT: 8
- OTHER -> CUSTOMER_SUPPORT: 7
- CUSTOMER_SUPPORT -> BILLING_PAYMENT: 6
- BILLING_PAYMENT -> CUSTOMER_SUPPORT: 5
- NETWORK_CONNECTIVITY -> SWITCHING_RETENTION: 4
- NETWORK_CONNECTIVITY -> WIFI_ROUTER: 4

## Finding

`CUSTOMER_SUPPORT` has several major semantic overlaps. This suggests that the next work should investigate class boundaries and feature representation rather than blindly deleting data.

---

# 5. Dataset Quality Audit

The dataset audit found:

- 330 evaluation examples
- 23 very-short examples in the initial audit
- 0 duplicate texts
- 28 OTHER examples
- 42 NETWORK_CONNECTIVITY examples
- 96 CUSTOMER_SUPPORT examples

Examples of short/ambiguous data included:

- `Verizon sucks`
- `Fuck Verizon`
- `DM you what?`
- `Billing issues`
- `contacting billing now`
- `have a question`
- `need help with my fios`
- `My internet is not working.`

## Finding

Short text is not automatically bad data. Some short examples contain clear intent signals, while others are genuinely ambiguous.

Therefore, blanket deletion is not justified.

---

# 6. Short-Text Removal Experiment

Three datasets were compared:

1. CURRENT
2. SHORT_REMOVED
3. SHORT_PLUS_OTHER

Results:

| Experiment | Dataset Size | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| CURRENT | 330 | 0.6879 | 0.6795 | 0.6850 |
| SHORT_REMOVED | 327 | 0.7037 | **0.6939** | 0.7008 |
| SHORT_PLUS_OTHER | 317 | 0.7101 | 0.6739 | 0.7050 |

## Finding

Removing short examples improved Macro F1 by about +0.0144 in this side experiment.

However:
- dataset size changed
- class distribution changed
- variance remained
- some removed examples were legitimate and informative

Therefore this was not promoted to production.

The SHORT_PLUS_OTHER variant had higher accuracy but lower Macro F1 than the current baseline, so it should not be selected on accuracy alone.

---

# 7. Short-Text Confusion Inspection

The short-text inspection showed 14 removed examples.

Distribution:

| Class | Removed |
|---|---:|
| CUSTOMER_SUPPORT | 6 |
| OTHER | 4 |
| BILLING_PAYMENT | 2 |
| WIFI_ROUTER | 1 |
| NETWORK_CONNECTIVITY | 1 |

Examples:

- `@115725 DM you what?`
- `@VerizonSupport Router netgear DM200`
- `Verizon sucks!!!`
- `@115725 you're not down`
- `@VerizonSupport contacting billing now`
- `@VerizonSupport Billing issues`
- `@VerizonSupport have a question`

The inspection showed that some short examples are clearly useful.

Conclusion: do not automatically delete all short examples.

---

# 8. Targeted OTHER Cleanup Experiment

Four ambiguous OTHER examples were removed:

1. `@VerizonSupport Is anyone else having trouble with verizon`
2. `@VerizonSupport need help with my fios`
3. `@VerizonSupport Need help with an order`
4. `@115990 been on the phone all day with Verizon.. Incompetent.. Need help`

Results:

| Experiment | Dataset Size | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| CURRENT | 330 | 0.687879 | 0.679516 | 0.684972 |
| TARGETED_OTHER_REMOVED | 326 | 0.705594 | 0.680257 | 0.703253 |

Changes:
- Accuracy: +0.017716
- Macro F1: +0.000742
- Weighted F1: +0.018281

## Finding

The accuracy increase is noticeable, but Macro F1 improved by only +0.000742.

That is too small to justify changing the production dataset based on this experiment alone.

Conclusion: keep the four examples for now.

---

# 9. Overall Findings So Far

## Supported by experiments

- Word bigrams are useful when combined with sublinear TF and `min_df=2`.
- Balanced class weights are important for this dataset.
- `CUSTOMER_SUPPORT` is involved in several major confusion boundaries.
- Some short examples are ambiguous.
- Some short examples are clearly useful.
- Removing selected OTHER examples did not materially improve Macro F1.

## Not justified yet

- deleting all short texts
- deleting the ambiguous OTHER examples
- modifying production training code
- selecting a model based on accuracy alone

---

# 10. Current Working Baseline

```python
TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)
```

Classifier:

```python
LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
)
```

Reference performance:

- Accuracy: approximately 0.6879
- Macro F1: approximately 0.6797
- Weighted F1: approximately 0.6864

Macro F1 is the primary comparison metric.

---

# 11. NEXT EXPERIMENT: Character TF-IDF

The next planned side experiment compares:

1. Current best word TF-IDF
2. Character TF-IDF
3. Character TF-IDF with `min_df=2`
4. Combined word + character TF-IDF

Character configuration:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=1,
    sublinear_tf=True,
)
```

Second character configuration:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    sublinear_tf=True,
)
```

Combined representation:
- word n-grams `(1,2)`
- character n-grams `(2,5)`
- sublinear TF

## Why test character TF-IDF?

The dataset contains noisy support-style text:
- repeated characters such as `sssllllllooooooowwwwwww`
- spelling/noise variations
- short informal phrases
- usernames and URLs
- fragmented conversational text

Character n-grams may capture patterns that complete-word features miss.

This is only a hypothesis. The experiment must determine whether it actually helps.

---

# 12. Exact Next Action

Run:

```powershell
python src/models/test_char_tfidf_experiment.py
```

Keep it as a side experiment.

Do not modify production training code yet.

---

# 13. Decision Rules

### If character TF-IDF improves Macro F1 clearly
Inspect its confusion matrix and determine which boundaries improved.

### If word + character improves Macro F1
Perform deeper out-of-fold confusion inspection before considering production changes.

### If character TF-IDF performs worse
Discard this direction.

### If differences are tiny
Treat the configurations as effectively similar and move to another controlled experiment.

The model should not be changed merely because a technique sounds sophisticated.

---

# 14. Project Rules

1. Do not modify production training code during side experiments.
2. Do not modify the production dataset based on one experiment.
3. Treat project source code as the implementation source of truth.
4. Separate verified results from hypotheses.
5. Compare Macro F1 before accuracy.
6. Inspect confusion matrices before promoting a model.
7. Do not assume short examples are bad.
8. Do not assume OTHER examples should be deleted because its F1 is low.
9. Keep cross-validation reproducible.
10. Record every experiment and result in this file.

---

# 15. Experiment Status

## Completed
- [x] Baseline word TF-IDF comparison
- [x] Best 6-class configuration identified
- [x] 6-class confusion inspection
- [x] Dataset quality audit
- [x] Short-text removal experiment
- [x] Short-text confusion inspection
- [x] Targeted OTHER cleanup experiment

## Current
- [ ] Character TF-IDF experiment

## After Character TF-IDF
- [ ] Inspect best character/combined configuration
- [ ] Compare confusion matrices
- [ ] Check CUSTOMER_SUPPORT boundaries
- [ ] Decide whether another feature experiment is justified
- [ ] Only then consider production-model changes

---

# 16. Current Working Conclusion

The strongest evidence so far supports the improved word-level TF-IDF configuration:

**word unigrams + bigrams + sublinear TF + min_df=2 + balanced class weights**

The data-cleaning experiments have not provided enough Macro F1 evidence to justify production dataset changes.

The major unresolved issue is class overlap, especially involving `CUSTOMER_SUPPORT`.

The next controlled experiment is therefore character-level TF-IDF, followed by a word + character combination.

No production changes should be made until those results are inspected.

Experiment 7 conclusion

I would record the conclusion as:

Word + Character TF-IDF improved 5-fold out-of-fold Macro F1 from 0.6797 to 0.6901 (+0.0104). The improvement was primarily driven by CUSTOMER_SUPPORT, with smaller gains in BILLING_PAYMENT and WIFI_ROUTER. However, NETWORK_CONNECTIVITY and OTHER became less reliable, particularly due to increased confusion between NETWORK_CONNECTIVITY and CUSTOMER_SUPPORT. Therefore, Word + Character TF-IDF is a promising candidate, but it should not yet be selected as the final classifier without further robustness and classifier comparisons.

And importantly:

Do not modify the production model yet.

We're still experimenting.

## Experiment 7: Full Word vs Word+Character TF-IDF Error Analysis

### Hypothesis
Character-level TF-IDF features may improve intent classification by capturing
patterns that word-level features miss, especially variations in spelling,
phrasing, and short text patterns.

### Setup
- Evaluation dataset: 330 manually labeled examples
- Classes: 6
- Validation: 5-fold stratified out-of-fold predictions
- Classifier: Logistic Regression with class balancing
- Word model:
  - TF-IDF word n-grams
  - ngram_range=(1,2)
  - min_df=2
  - sublinear_tf=True
- Word + character model:
  - Same word features
  - Character TF-IDF n-grams
  - ngram_range=(2,5)
  - min_df=1
  - sublinear_tf=True
- Metric: Macro F1

### Results

| Model | Macro F1 |
|---|---:|
| Word bigrams | 0.6797 |
| Word + character TF-IDF | 0.6901 |
| Change | +0.0104 |

### Per-Class Error Transitions

| Class | Corrected | Broken | Net Change |
|---|---:|---:|---:|
| BILLING_PAYMENT | 3 | 1 | +2 |
| CUSTOMER_SUPPORT | 10 | 1 | +9 |
| NETWORK_CONNECTIVITY | 1 | 4 | -3 |
| OTHER | 0 | 3 | -3 |
| SWITCHING_RETENTION | 1 | 1 | 0 |
| WIFI_ROUTER | 1 | 0 | +1 |

### Key Findings
Word + character TF-IDF improved overall Macro F1 by 0.0104.

The largest improvement occurred for CUSTOMER_SUPPORT, with 10 examples
corrected and only 1 example becoming incorrect.

However, NETWORK_CONNECTIVITY became less reliable. The largest confusion
change was:

NETWORK_CONNECTIVITY -> CUSTOMER_SUPPORT: 2 -> 8

OTHER also became weaker, with:

OTHER -> CUSTOMER_SUPPORT: 7 -> 10

This indicates that character features provide useful additional signal but
do not improve every intent boundary.

### Conclusion
Word + character TF-IDF is a stronger candidate than word-only TF-IDF based
on this experiment, but it should not yet be selected as the final classifier.
Further experiments should test character n-gram configurations and alternative
classifiers before freezing the model.

### Production Impact
No production dataset or production training code was modified.

## Experiment 8: Character N-Gram Tuning

### Hypothesis
The character n-gram range affects the usefulness of character-level
features. Shorter character patterns may capture useful variations in
customer messages without introducing unnecessary feature noise.

### Setup
- Evaluation dataset: 330 manually labeled examples
- Classes: 6
- Validation: 5-fold stratified cross-validation
- Classifier: Logistic Regression with class balancing
- Word features fixed:
  - TF-IDF word n-grams: (1,2)
  - min_df=2
  - sublinear_tf=True
- Character features varied:
  - min_df=1
  - sublinear_tf=True
- Primary metric: Macro F1

### Results

| Character n-gram range | Mean Macro F1 | Std |
|---|---:|---:|
| (2,4) | 0.7088 | 0.0517 |
| (2,5) | 0.7052 | 0.0468 |
| (2,6) | 0.6968 | 0.0523 |
| (3,6) | 0.6921 | 0.0448 |
| (3,5) | 0.6919 | 0.0515 |

### Key Finding
The (2,4) character configuration achieved the highest Macro F1 at 0.7088.
It outperformed the other character ranges tested in the same experiment.

The results suggest that shorter character patterns provide useful signal
for this dataset, while extending the character range to 5 or 6 characters
did not improve performance.

### Conclusion
Use CHAR_2_4 as the current best character-feature configuration for the
next experiment.

However, the model should not yet be frozen because classifier choice has
not been evaluated systematically.

### Caveat
The CHAR_2_5 score in this experiment (0.7052) differs from the CHAR_2_5
result reported in Experiment 7 (0.6901). Therefore, comparisons across
those two experiments should be treated cautiously. The configuration
ranking within Experiment 8 is the more reliable comparison because all
configurations use the same experimental procedure.

### Production Impact
No production dataset or production training code was modified.

## Experiment 9: Classifier Comparison

### Hypothesis
The choice of linear classifier may affect performance even when the TF-IDF
feature representation is fixed.

### Setup
- Evaluation dataset: 330 manually labeled examples
- Classes: 6
- Validation: 5-fold stratified cross-validation
- Word TF-IDF:
  - ngram_range=(1,2)
  - min_df=2
  - sublinear_tf=True
- Character TF-IDF:
  - ngram_range=(2,4)
  - min_df=1
  - sublinear_tf=True
- Combined word + character features
- Primary metric: Macro F1

### Classifiers Tested
1. Logistic Regression
2. Linear SVM
3. Complement Naive Bayes

### Results

| Classifier | Mean Macro F1 | Std |
|---|---:|---:|
| Linear SVM | 0.7100 | 0.0568 |
| Logistic Regression | 0.7088 | 0.0517 |
| Complement Naive Bayes | 0.6360 | 0.0478 |

### Key Findings
Linear SVM achieved the highest mean Macro F1 at 0.7100.

However, the difference between Linear SVM and Logistic Regression was
only 0.0012, which is small relative to the observed fold variability.

Complement Naive Bayes performed substantially worse at 0.6360.

The experiment therefore suggests that the feature representation has a
larger effect on performance than the choice between the two strongest
linear classifiers tested.

### Conclusion
Linear SVM is the current best-performing classifier, but its advantage
over Logistic Regression is too small to treat as a decisive improvement.

Use Linear SVM as the current candidate for the next robustness experiment.

### Production Impact
No production dataset or production training code was modified.

## Experiment 10: Final Classifier Robustness

### Hypothesis
The small advantage of Linear SVM over Logistic Regression observed in
Experiment 9 may be caused by a particular train/test split. Repeated
stratified cross-validation can test whether the difference is reasonably
stable.

### Setup
- Evaluation dataset: 330 manually labeled examples
- Classes: 6
- Validation: 5-fold stratified cross-validation
- Repeated 3 times
- Total evaluation folds: 15
- Word TF-IDF:
  - ngram_range=(1,2)
  - min_df=2
  - sublinear_tf=True
- Character TF-IDF:
  - ngram_range=(2,4)
  - min_df=1
  - sublinear_tf=True
- Classifiers:
  - Linear SVM
  - Logistic Regression
- Primary metric: Macro F1

### Results

| Classifier | Mean Macro F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| Linear SVM | 0.7216 | 0.0487 | 0.6104 | 0.8213 |
| Logistic Regression | 0.7168 | 0.0578 | 0.6266 | 0.8223 |

### Key Findings
Linear SVM achieved the highest mean Macro F1 at 0.7216.

Logistic Regression achieved a mean Macro F1 of 0.7168.

The mean difference was +0.0048 in favor of Linear SVM.

Linear SVM also showed lower fold-to-fold variability:
0.0487 standard deviation compared with 0.0578 for Logistic Regression.

### Conclusion
Linear SVM is the current classifier choice for the intent classification
component.

The result is reasonably consistent across repeated splits, although the
difference from Logistic Regression is modest.

The repeated cross-validation score should be treated as a model-selection
estimate rather than a final held-out evaluation score.

### Production Impact
No production dataset or production training code was modified.