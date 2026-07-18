"""Shared environment/LLM bootstrap used by agent.py, extract.py and router.py."""

import os

from dotenv import load_dotenv
from netfree_unstrict_ssl import unstrict_ssl
from llama_index.llms.google_genai import GoogleGenAI

GEMINI_MODEL = "models/gemini-2.0-flash-lite"

_initialized = False


def init_env() -> None:
    global _initialized
    if _initialized:
        return
    load_dotenv()
    unstrict_ssl()
    _initialized = True


def get_env_var(name: str) -> str:
    init_env()
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is missing from the environment")
    return value


def get_llm(temperature: float = 0.1) -> GoogleGenAI:
    return GoogleGenAI(
        model=GEMINI_MODEL,
        api_key=get_env_var("GEMINI_API_KEY"),
        temperature=temperature,
    )
