
import os
from unittest.mock import patch

import pytest

from src.rag.llm_generator import LLMGenerator


def test_missing_api_key_raises_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY is missing",
        ):
            LLMGenerator()


def test_empty_prompt_raises_error():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        generator = LLMGenerator()

        with pytest.raises(
            ValueError,
            match="Prompt must not be empty",
        ):
            generator.generate("   ")
