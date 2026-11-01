
import pandas as pd


def format_retrieved_context(results: pd.DataFrame) -> str:
    """Format retrieved support cases into context for an LLM."""

    if results.empty:
        return "No relevant historical support cases were found."

    context_parts = []

    for index, row in enumerate(results.itertuples(index=False), start=1):
        context_parts.append(
            f"""Case {index}
Customer message: {row.customer_message}
Support response: {row.support_response}"""
        )

    return "\n\n".join(context_parts)
