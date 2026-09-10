"""
llm_config.py

Two model tiers:

  get_worker_llm()     → GPT-4.1-mini on Azure   (worker agents, streaming)
  get_extraction_llm() → GPT-4.1-mini on Azure   (structured output, streaming=False)
  get_routing_llm()    → Gemini Flash 2.5 Lite    (orchestrator routing)

Orchestrator uses Gemini via service account (GCP_SA_BASE64).
Falls back to GPT-4.1-mini if LLM_PROVIDER != "gemini" or config missing.

Required .env for Gemini orchestrator:
  LLM_PROVIDER=gemini
  LLM_MODEL=gemini-2.5-flash-lite
  GCP_PROJECT_ID=precallautomation
  GCP_LOCATION=global
  GEMINI_THINKING_LEVEL=low
  GEMINI_THINKING_BUDGET=0
  GCP_SA_BASE64=<base64 encoded service account JSON>
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Final


def _load_dotenv():
    try:
        for dot in [Path(__file__).parents[2] / ".env", Path.cwd() / ".env"]:
            if not dot.exists():
                continue
            for line in dot.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            break
    except Exception:
        pass


_load_dotenv()


class Config:
    # Azure OpenAI — worker agents
    AZURE_OPENAI_API_KEY: Final[str] = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: Final[str] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    OPENAI_API_VERSION: Final[str] = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
    WORKER_DEPLOYMENT: Final[str] = os.getenv("WORKER_DEPLOYMENT", "gpt-5.4-nano")
    WORKER_TEMPERATURE: Final[float] = float(os.getenv("WORKER_TEMPERATURE") or "0")

    # Gemini — orchestrator
    LLM_PROVIDER: Final[str] = os.getenv("LLM_PROVIDER", "")
    LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    GCP_PROJECT_ID: Final[str] = os.getenv("GCP_PROJECT_ID", "")
    GCP_LOCATION: Final[str] = os.getenv("GCP_LOCATION", "global")
    GEMINI_THINKING_LEVEL: Final[str] = os.getenv("GEMINI_THINKING_LEVEL", "low")
    # GEMINI_THINKING_BUDGET: Final[int] = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
    GEMINI_THINKING_BUDGET: Final[int] = int(os.getenv("GEMINI_THINKING_BUDGET") or "0")
    GCP_SA_BASE64: Final[str] = os.getenv("GCP_SA_BASE64", "")

    # Salesforce OAuth
    SF_CLIENT_ID: Final[str] = os.getenv("SF_CLIENT_ID", "")
    SF_CLIENT_SECRET: Final[str] = os.getenv("SF_CLIENT_SECRET", "")
    SF_REFRESH_TOKEN: Final[str] = os.getenv("SF_REFRESH_TOKEN", "")
    SF_TOKEN_URL: Final[str] = os.getenv("SF_TOKEN_URL", "https://login.salesforce.com/services/oauth2/token")
    SF_API_VERSION: Final[str] = os.getenv("SF_API_VERSION", "v60.0")
    SF_INSTANCE_URL: Final[str] = os.getenv("SF_INSTANCE_URL", "")

    # LangSmith tracing
    LANGCHAIN_API_KEY: Final[str] = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: Final[str] = os.getenv(
        "LANGCHAIN_PROJECT", "conversation-agent-cigna-member-sdo-langgraph"
    )


@lru_cache(maxsize=4)
def get_worker_llm(max_tokens: int = 512):
    """
    Worker agent LLM — GPT-4.1-mini on Azure.
    streaming=True required for stream_mode="messages" to work in LangGraph.
    Singleton via lru_cache — one instance per process.
    """
    # Lazy import: prevents import-time failure when langchain_openai is absent
    # (e.g. running tests without the full dependency set installed).
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        azure_deployment=Config.WORKER_DEPLOYMENT,
        temperature=Config.WORKER_TEMPERATURE,
        max_tokens=max_tokens,
        streaming=False,
    )


@lru_cache(maxsize=1)
def get_extraction_llm():
    """
    Structured-output LLM — GPT-4.1-mini on Azure with streaming=False.
    Used for all .with_structured_output(WorkerResult) calls.
    Singleton via lru_cache — one instance per process.
    """
    # Lazy import: prevents import-time failure when langchain_openai is absent.
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        azure_deployment=Config.WORKER_DEPLOYMENT,
        temperature=Config.WORKER_TEMPERATURE,
        max_tokens=300,
        timeout=30.0,  # hard per-request ceiling
        max_retries=1,  # don't let the SDK silently backoff to 7s
        streaming=False,
    )


@lru_cache(maxsize=1)
def get_follow_up_llm():
    """
    Follow-up agent LLM — same model but higher max_tokens.
    Used by follow_up_agent for combined extraction + answer generation.
    """
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        azure_deployment=Config.WORKER_DEPLOYMENT,
        temperature=Config.WORKER_TEMPERATURE,
        max_tokens=400,
        streaming=False,
    )


@lru_cache(maxsize=1)
def get_generation_llm():
    """LLM 2 — Gemini for recovery/utterance generation. Cached singleton.
    Falls back to get_extraction_llm() if Gemini is unavailable.
    """
    try:
        return get_gemini_llm()
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "get_generation_llm: Gemini unavailable — falling back to extraction LLM"
        )
        return get_extraction_llm()


@lru_cache(maxsize=1)
def get_routing_llm():
    """
    Orchestrator LLM — Gemini Flash 2.5 Lite via GCP service account.
    Fast and cheap for routing decisions (~200-300ms).
    Falls back to GPT-4.1-mini if Gemini is not configured or credentials are invalid.
    Cached: the Gemini path re-decodes GCP_SA_BASE64 and re-creates credentials on
    every call without caching — expensive and unnecessary.
    """
    # if Config.LLM_PROVIDER == "gemini":
    #     try:
    #         return get_gemini_llm()
    #     except ValueError as exc:
    #         import logging

    #         logging.getLogger(__name__).warning(
    #             "get_routing_llm: Gemini config invalid (%s) — falling back to GPT", exc
    #         )
    # return get_worker_llm()
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        azure_deployment=Config.WORKER_DEPLOYMENT,
        temperature=0.3,
        max_tokens=300,
        streaming=False,
    )


# ── Gemini builder ────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_gemini_llm():
    """
    Build ChatGoogleGenerativeAI using GCP service account credentials.

    Uses langchain_google_genai (not langchain_google_vertexai) — confirmed
    to work with thinking_budget on Gemini 2.5 Flash Lite.

    Thinking config:
      gemini-3.x models → thinking_level (low | medium | high)
      gemini-2.x models → thinking_budget (int, 0 = no thinking)
    """
    # Lazy imports: prevent import-time failures when google-auth or
    # langchain_google_genai are absent (non-Gemini deployments).
    import base64
    import json

    from google.oauth2 import service_account
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Decode and parse service account JSON
    encoded_sa = Config.GCP_SA_BASE64
    if not encoded_sa:
        raise ValueError("GCP_SA_BASE64 is not set. Provide a base64-encoded GCP service account JSON.")

    try:
        decoded_str = base64.b64decode(encoded_sa).decode()
        credentials_info = json.loads(decoded_str)
    except Exception as e:
        raise ValueError(f"Invalid GCP_SA_BASE64: {e}")

    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    model_name = Config.LLM_MODEL
    thinking_level = Config.GEMINI_THINKING_LEVEL
    thinking_budget = Config.GEMINI_THINKING_BUDGET

    gemini_kwargs = {
        "model": model_name,
        "project": Config.GCP_PROJECT_ID,
        "location": Config.GCP_LOCATION,
        "credentials": credentials,
        "temperature": 0.3,
    }

    # Thinking config differs between gemini-2.x and gemini-3.x
    if "gemini-3" in model_name:
        gemini_kwargs["thinking_level"] = thinking_level
    else:
        gemini_kwargs["thinking_budget"] = int(thinking_budget)

    llm = ChatGoogleGenerativeAI(**gemini_kwargs)

    # Metadata visible in LangSmith traces
    llm._experiment_metadata = {
        "provider": "gemini",
        "model": model_name,
    }

    return llm
