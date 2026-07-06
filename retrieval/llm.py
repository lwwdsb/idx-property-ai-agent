"""Reserved LLM slot for the verifier's reasoning layer (Week 7, part 3).

The DETERMINISTIC gate is the safety floor and never depends on this. When an
LLM key is configured, `explain_verdict` adds a natural-language, nuance-aware
explanation on TOP of the deterministic verdict (e.g. "above median, but the only
one with a pool + view, so the premium may be justified"). Empty key => returns
None and callers fall back to the plain deterministic verdict.
"""
import json
import urllib.request

from common import load_env


def llm_available() -> bool:
    env = load_env()
    return bool(env.get("LLM_API_KEY", "").strip())


def explain_verdict(context: dict) -> str | None:
    """context: listing summary + comp stats + deterministic verdict.
    Returns a short reasoned explanation, or None if no LLM configured / on error."""
    env = load_env()
    key = env.get("LLM_API_KEY", "").strip()
    if not key:
        return None  # reserved slot — deterministic verdict stands on its own
    base = env.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = env.get("LLM_MODEL", "gpt-4o-mini")
    prompt = (
        "You are a real-estate pricing reviewer. Given a listing and recent comparable "
        "sales, in ONE or TWO sentences explain whether the asking price looks justified. "
        "Ground every claim in the numbers provided; do not invent facts.\n\n"
        + json.dumps(context, ensure_ascii=False)
    )
    body = json.dumps({
        "model": model, "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None  # 乙: reasoning failure must not break the verdict
