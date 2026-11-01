
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMGenerator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your project .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text.strip()
