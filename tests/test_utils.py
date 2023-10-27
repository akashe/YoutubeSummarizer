# 3 for all functions
# writing edge cases
import os


def test_open_ai_key():
    assert "OPENAI_API_KEY" in os.environ


