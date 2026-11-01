Project:
Hiver SDE Intern Take-Home

Deadline:
17 September 2026

Goal:
Build an AI support agent for one brand from the
Customer Support on Twitter dataset.

Agent requirements:
1. Intent classification
2. Historically grounded reply generation
3. Auto-handle vs escalation decision

Required evaluation:
- 150–250 hand-labelled golden examples
- trivial baseline
- simple baseline
- automated metrics
- LLM-as-judge
- human vs LLM judge agreement
- failure analysis
- misleading headline number
- decision log

Current dataset:
2,811,774 rows
7 columns
702,777 unique authors

Current findings:
inbound=True  → 1,537,843
inbound=False → 1,273,931

Top support accounts discovered:
AmazonHelp
AppleSupport
Uber_Support
SpotifyCares
Delta
...

Current understanding:
Support-account tweets appear as inbound=False.
Conversation relationships are represented through
response_tweet_id and in_response_to_tweet_id.

Current phase:
Dataset exploration.

Current unresolved question:
Which brand should we choose?