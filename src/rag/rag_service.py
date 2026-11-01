
from src.retrieval.format_context import format_retrieved_context
from src.rag.prompt_template import build_rag_prompt
from src.rag.llm_generator import LLMGenerator


class RAGService:
    def __init__(self, retrieval_service=None, llm_generator=None):
        self.retrieval_service = retrieval_service
        self.llm_generator = llm_generator

    def _prepare_prompt_and_sources(
        self,
        customer_query: str,
        top_k: int = 5,
    ) -> tuple[str, list[dict[str, str]]]:
        if not isinstance(customer_query, str) or not customer_query.strip():
            raise ValueError("Customer query must not be empty.")

        if self.retrieval_service is None:
            from src.retrieval.retrieval_service import RetrievalService
            self.retrieval_service = RetrievalService()

        results = self.retrieval_service.retrieve_cases(
            customer_query,
            top_k=top_k,
        )


        sources = []
        for rank, row in enumerate(results.itertuples       (index=False), start=1):
            sources.append(
                {
                    "rank": rank,
                    "customer_message": str(row.customer_message),
                    "support_response": str(row.support_response),
                    "rrf_score": float(row.similarity),
                }
        )


        context = format_retrieved_context(results)

        prompt = build_rag_prompt(
            customer_query=customer_query,
            retrieved_context=context,
        )

        return prompt, sources

    def prepare_prompt(self, customer_query: str, top_k: int = 5) -> str:
        prompt, _ = self._prepare_prompt_and_sources(customer_query, top_k)
        return prompt

    def answer(self, customer_query: str, top_k: int = 5) -> str:
        prompt = self.prepare_prompt(customer_query, top_k=top_k)

        if self.llm_generator is None:
            self.llm_generator = LLMGenerator()

        return self.llm_generator.generate(prompt)

    def answer_with_sources(
        self,
        customer_query: str,
        top_k: int = 5,
    ) -> dict:
        prompt, sources = self._prepare_prompt_and_sources(
            customer_query,
            top_k=top_k,
        )

        if self.llm_generator is None:
            self.llm_generator = LLMGenerator()

        answer = self.llm_generator.generate(prompt)

        return {
            "answer": answer,
            "sources": sources,
        }
