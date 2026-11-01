
from src.rag.rag_service import RAGService
from src.rag.llm_generator import LLMGenerator


def main():
    rag_service = RAGService()
    generator = LLMGenerator()

    test_queries = [
        "My internet keeps disconnecting. What should I do?",
        "Why has my Verizon bill suddenly increased?",
        "I want to cancel my Verizon service. What are my options?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print("CUSTOMER QUESTION:")
        print(query)

        prompt = rag_service.prepare_prompt(query, top_k=3)
        answer = generator.generate(prompt)

        print("\nAI SUPPORT RESPONSE:")
        print(answer)


if __name__ == "__main__":
    main()
