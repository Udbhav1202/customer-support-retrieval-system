# Project:
## Hiver SDE Intern Take-Home

## Deadline:

17 September 2026

## Goal:

Build an AI support agent for one brand from the
Customer Support on Twitter dataset.

## Agent requirements:

1. Intent classification
2. Historically grounded reply generation
3. Auto-handle vs escalation decision

## Required evaluation:

- 150–250 hand-labelled golden examples
- trivial baseline
- simple baseline
- automated metrics
- LLM-as-judge
- human vs LLM judge agreement
- failure analysis
- misleading headline number
- decision log

## Current dataset:

2,811,774 rows
7 columns
702,777 unique authors

## Current findings:

inbound=True  → 1,537,843
inbound=False → 1,273,931

## Top support accounts discovered:

AmazonHelp
AppleSupport
Uber_Support
SpotifyCares
Delta
...

## Current understanding:

Support-account tweets appear as inbound=False.
Conversation relationships are represented through
response_tweet_id and in_response_to_tweet_id.

## Current phase:

Dataset exploration.

## Current unresolved question:

Which brand should we choose?

## Current Findings

- Dataset contains 2,811,774 tweets.
- Dataset contains 7 columns.
- There are 702,777 unique authors.
- 1,537,843 tweets have `inbound=True`.
- 1,273,931 tweets have `inbound=False`.
- Major support accounts appear as outbound accounts.
- Conversations are connected using `response_tweet_id` and
  `in_response_to_tweet_id`.
- Conversations can branch rather than being strictly linear.
- `tweet_id` should not be assumed to represent chronological order.
- `created_at` should be considered for chronological ordering.

## Current Phase

Dataset exploration and conversation structure analysis.

## Next Step

Analyze candidate brands based on the amount and quality of their
customer-support conversations.

## Confirmed Conversation Behavior

- A tweet can respond to another tweet.
- A tweet can have multiple responses.
- Response relationships can be represented in both directions.
- Conversations may form trees rather than simple linear chains.
- Tweet ID order does not represent conversation order.

## Current Phase
Brand selected: VerizonSupport.

Dataset exploration and initial conversation reconstruction are complete.

## Key Brand Selection Findings
- VerizonSupport selected as the target brand.
- 17,966 support tweets.
- 17,805 linked customer messages.
- 99.98% of linked customer messages pass the current English heuristic.
- Average sampled conversation length: 4.63 tweets.
- 253/500 sampled interactions had at least 4 tweets.
- Qualitative inspection showed troubleshooting, diagnostic questioning,
  escalation, remote fixes, and outage handling.

## Next Step
Build a clean conversation-level dataset for VerizonSupport.

## Current Phase
Brand selection and initial conversation reconstruction are complete.

A clean retrieval dataset has now been built for VerizonSupport.

## Retrieval Dataset

Original retrieval cases:
- 23,859 cases
- 8,358 conversations

Cleaning:
- Removed 2,130 cases from conversations longer than 30 messages.
- Removed 456 customer messages shorter than 20 characters.
- Removed 5 exact customer/support duplicate pairs.

Final retrieval dataset:
- 21,268 cases
- 8,299 conversations
- Mean customer message length: 104.77 characters
- Median customer message length: 102 characters
- Maximum conversation length: 30 messages

The original retrieval dataset was preserved unchanged.

## Important Caveat

The cleaning thresholds are provisional engineering choices.
They have not yet been validated through retrieval evaluation.

## Next Step

Build and evaluate a retrieval baseline using the cleaned
VerizonSupport dataset.

The initial retrieval experiment should establish whether
historically similar customer messages retrieve useful support
responses before adding LLM-based response generation.