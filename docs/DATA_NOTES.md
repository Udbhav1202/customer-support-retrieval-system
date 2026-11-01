## Conversation Structure

The dataset represents conversations using tweet relationships rather
than simply storing one customer message and one support response per row.

`in_response_to_tweet_id` identifies the tweet that the current tweet
responds to.

`response_tweet_id` identifies one or more tweets responding to the
current tweet.

A conversation can therefore branch into multiple responses rather
than always forming a simple linear sequence.

### Example

For one AmazonHelp conversation:

272 (customer)
→ 269 (AmazonHelp)
→ 270 (customer)
→ 271 (customer)
→ 273 (AmazonHelp)
→ 274 (customer)
→ 275 (AmazonHelp)

Tweet 269 responds to tweet 272, while tweets 270 and 271 both respond
to tweet 269.

### Important observation

`tweet_id` is not sufficient to determine conversation order.
Conversation relationships should be reconstructed using the response
ID fields, while `created_at` can be used for chronological ordering.

## Example Conversation Reconstruction

A sample AmazonHelp conversation demonstrates that the dataset can
represent branching conversations.

Tweet 272 is a customer tweet.

Tweet 269 is an AmazonHelp response to tweet 272 because:
- `269.in_response_to_tweet_id = 272`
- `272.response_tweet_id = 269`

Tweet 269 has two responses:
- tweet 270
- tweet 271

Both are from the same customer and have
`in_response_to_tweet_id = 269`.

Therefore, one tweet can have multiple response tweets and a
conversation cannot be assumed to be a simple one-to-one sequence.

The response ID fields should be treated as relationships between
tweets.

## Timestamp Handling

The raw `created_at` field is parsed once when the dataset is loaded
using the expected Twitter timestamp format.

Invalid timestamps are converted to missing values using
`errors="coerce"`.

In the initial sample of 100 VerizonSupport conversations:
- 0 conversations contained missing timestamps after parsing.

Timestamp quality will be checked again on the larger dataset before
finalizing the data pipeline.

## Cross-Brand Noise

In a sample of 1,000 initial VerizonSupport customer messages,
23 messages (2.3%) contained explicit mentions of other support
brands such as Comcast, Spectrum, AT&T, and Cox.

The messages were not uniformly irrelevant. They included:
- complaints about competitor service
- customers considering switching to Verizon
- comparisons between Verizon and competitors
- competitor service problems where Verizon was mentioned
- possible Verizon sales/retention opportunities

### Important Observation

A competitor-brand mention should not automatically be treated
as noise.

Some messages may represent valid Verizon-related intents such
as switching, pricing, or sales inquiries, while others are
primarily about a competitor.

Therefore, competitor mentions should be inspected as part of
intent discovery rather than removed using a simple keyword filter.

### Current Decision

Do not apply a competitor-brand exclusion rule yet.

Further inspection is required before defining the cleaning rule
for intent discovery.