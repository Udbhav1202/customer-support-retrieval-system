## Experiment: 100 VerizonSupport Conversation Structure

### Hypothesis
VerizonSupport conversations contain enough multi-turn interaction
to provide useful historical evidence for intent classification,
response generation, and escalation decisions.

### Method
Sampled 100 conversation roots from the 8,358 VerizonSupport
conversation roots and reconstructed each conversation using the
response relationships.

Measured:
- messages per conversation
- customer messages
- support messages
- missing timestamps

### Results
- Conversations tested: 100
- Average messages per conversation: 5.49
- Minimum messages: 2
- Maximum messages: 36
- Average customer messages: 3.22
- Average support messages: 2.27
- Conversations with missing timestamps: 0

### Conclusion
The sample contains substantial multi-turn interaction and
conversation-length variation. This supports using conversation
history as evidence for the support agent. However, the sample is
too small to make dataset-wide claims about timestamp quality or
conversation structure.

### Next Step
Create the final VerizonSupport conversation dataset and inspect
the distribution of conversation lengths and message composition
on a larger sample.

## Experiment: Cross-Brand Mentions

### Hypothesis
Some VerizonSupport customer messages may contain references to
other telecom providers and could introduce noise into intent
discovery.

### Method
Checked 1,000 initial VerizonSupport customer messages for explicit
mentions of selected competitor support accounts:
Comcast, Spectrum, AT&T, Optimum, and Cox.

### Results
- Messages inspected: 1,000
- Messages containing competitor mentions: 23
- Percentage: 2.3%

### Qualitative Finding
The messages included competitor complaints, switching
considerations, price comparisons, and messages where a competitor
was responsible for the problem.

Therefore, competitor mentions are not necessarily irrelevant
noise.

### Conclusion
Do not remove competitor mentions using a simple keyword filter.
They require semantic inspection before establishing the cleaning
policy.

### Next Step
Inspect the broader initial-message sample and identify recurring
customer issue categories before defining the final intent taxonomy.

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