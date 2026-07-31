import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

# to hit portkey url we need a openai compitible url ie we are not using openai juust tricking langchain that we are using openai but using portkey 
#chatgroq dont give a way to override base url thats why we using openai
from APP.config import setting


# Production gateway config:
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @rag1/llama-3.1-8b-instant on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target


# GATEWAY_CONFIG = {   # to usee different feature of portkey 
#     "strategy": {"mode": "fallback"},
#     "cache": {"mode": "simple"},
#     "retry": {
#         "attempts": 2, # retry on2 attemps when status code is 429 or 503
#         "on_status_codes": [429, 503]
#     },
#     "targets": [ # load balancing
#         {"override_params": {"model": f"@{setting.GROQ_SLUG}/llama-3.3-70b-versatile"}},
#         {"override_params": {"model": f"@{setting.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
#     ]
# }

# we can directly define the config and use here we need to define it in the portkey congif at the dashboard
# and from there get the config id whcich we can use.

portkey_client = Portkey(
    api_key=setting.PORTKEY_API_KEY,
    config="pc-portke-ae3e04"
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=setting.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,  # using portkey 
        model=f"@{setting.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(  # header so we can see info in portkey dashboard
            api_key=setting.PORTKEY_API_KEY,
            config="pc-portke-ae3e04",
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"