"""LLM factories for the evaluation package (kept separate so src/self_rag.py is untouched)."""
import os
from langchain_groq import ChatGroq
from src.config import get_settings


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """LLM used by the Naive RAG baseline - same model as Self-RAG for a controlled comparison."""
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return ChatGroq(api_key=s.groq_api_key, model=s.groq_model, temperature=temperature)


def get_judge_llm() -> ChatGroq:
    """LLM used as the independent metric judge. Override with EVAL_JUDGE_MODEL to use a different model."""
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    model = os.getenv("EVAL_JUDGE_MODEL") or s.groq_model
    return ChatGroq(api_key=s.groq_api_key, model=model, temperature=0)
