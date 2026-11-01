## Decision: Select VerizonSupport as the target brand

### Context
The assignment requires selecting one brand from the Customer Support on Twitter dataset.

### Options considered
AmazonHelp, AppleSupport, VerizonSupport, TMobileHelp, XboxSupport,
GWRHelp, VirginTrains, idea_cares, Safaricom_Care, AirAsiaSupport.

### Evidence
VerizonSupport has:
- 17,966 support tweets
- 17,805 customer messages linked to support replies
- 99.98% of linked customer messages passing the English heuristic
- 4.63 average conversation length in a 500-interaction sample
- 253/500 sampled interactions containing at least 4 tweets

Qualitative inspection also showed useful troubleshooting sequences,
diagnostic questions, escalation to DM, remote fixes, and outage handling.

### Decision
Use VerizonSupport as the target brand.

### Rationale
VerizonSupport provides a strong balance between dataset size,
English-language coverage, multi-turn conversations, and useful
historical evidence of how customer issues were handled.

### Trade-off
AmazonHelp and AppleSupport provide substantially more raw data,
but VerizonSupport offers deeper conversations in the exploratory
sample, which is more useful for studying resolution behavior.

### Date
2026-09-11

## Decision: Create a Clean Retrieval Dataset

### Context
The initial retrieval dataset contained 23,859 customer/support
cases across 8,358 conversations.

Audit showed several problems:
- Extremely long conversations could contain unrelated later messages.
- Very short customer messages were often context-dependent and poor
  standalone retrieval queries.
- A small number of exact customer/support pairs were duplicated.

### Decision
Create a separate cleaned retrieval dataset while preserving the
original dataset.

Cleaning rules:
- Remove cases from conversations with more than 30 messages.
- Remove customer messages shorter than 20 characters.
- Remove exact duplicate customer/support pairs.
- Preserve conversation_id for later conversation-level splitting.

### Result
The cleaned dataset contains:
- 21,268 cases
- 8,299 conversations

### Trade-off
The cleaning rules may remove potentially useful examples.
Therefore, these thresholds are provisional and must be validated
through retrieval evaluation rather than treated as final production
rules.

### Date
2026-10-04