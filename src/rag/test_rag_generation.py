
from src.rag.rag_service import RAGService
from src.rag.llm_generator import LLMGenerator


def main():
    query = "My internet keeps disconnecting. What should I do?"

    # Retrieve relevant cases and prepare the prompt.
    rag_service = RAGService()
    prompt = rag_service.prepare_prompt(query, top_k=3)

    # Generate an answer using the LLM.
    generator = LLMGenerator()
    answer = generator.generate(prompt)

    print("\nCustomer question:")
    print(query)

    print("\nAI support response:")
    print(answer)


if __name__ == "__main__":
    main()
