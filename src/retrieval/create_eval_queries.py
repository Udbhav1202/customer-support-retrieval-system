import pandas as pd


queries = [
    # NETWORK_CONNECTIVITY
    ("my internet is extremely slow", "NETWORK_CONNECTIVITY"),
    ("my internet keeps disconnecting", "NETWORK_CONNECTIVITY"),
    ("my Fios connection is dropping", "NETWORK_CONNECTIVITY"),
    ("the internet has been terrible since yesterday", "NETWORK_CONNECTIVITY"),
    ("my connection keeps cutting out", "NETWORK_CONNECTIVITY"),

    # WIFI_ROUTER
    ("my wifi stopped working", "WIFI_ROUTER"),
    ("my router is not connecting", "WIFI_ROUTER"),
    ("the wifi is not working on any device", "WIFI_ROUTER"),
    ("my router has a red light", "WIFI_ROUTER"),
    ("I cannot connect to my wireless network", "WIFI_ROUTER"),

    # BILLING_PAYMENT
    ("I was charged too much", "BILLING_PAYMENT"),
    ("I need help with my bill", "BILLING_PAYMENT"),
    ("there is a problem with my payment", "BILLING_PAYMENT"),
    ("why is my bill so high", "BILLING_PAYMENT"),
    ("I was charged twice", "BILLING_PAYMENT"),

    # SWITCHING_RETENTION
    ("I want to cancel my Verizon service", "SWITCHING_RETENTION"),
    ("I want to switch to Verizon", "SWITCHING_RETENTION"),
    ("I'm thinking about leaving Verizon", "SWITCHING_RETENTION"),
    ("can I change my plan", "SWITCHING_RETENTION"),
    ("I want to move to another provider", "SWITCHING_RETENTION"),

    # CUSTOMER_SUPPORT
    ("I need help with my account", "CUSTOMER_SUPPORT"),
    ("I cannot access my account", "CUSTOMER_SUPPORT"),
    ("I need to speak with someone", "CUSTOMER_SUPPORT"),
    ("I need help with my Verizon account", "CUSTOMER_SUPPORT"),
    ("can someone help me", "CUSTOMER_SUPPORT"),

    # OTHER
    ("something is wrong", "OTHER"),
    ("I have a problem with my service", "OTHER"),
    ("this isn't working", "OTHER"),
    ("I don't know what is wrong", "OTHER"),
    ("I need assistance", "OTHER"),
]


df = pd.DataFrame(
    queries,
    columns=["query", "expected_intent"]
)

output_path = "data/processed/retrieval_eval_queries.csv"

df.to_csv(output_path, index=False)

print("=" * 80)
print("RETRIEVAL EVALUATION QUERIES")
print("=" * 80)

print(f"Total queries: {len(df)}")
print("\nQueries per intent:")
print(df["expected_intent"].value_counts())

print(f"\nSaved to: {output_path}")