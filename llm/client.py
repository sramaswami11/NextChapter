import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_FALLBACK = "Analysis complete — see your dashboard for the full breakdown."


def explain(system: str, user: str) -> str:
    model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    result = _ollama(system, user, model)
    if result:
        return result

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        result = _groq(system, user, groq_key)
        if result:
            return result

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        result = _anthropic(system, user, api_key)
        if result:
            return result

    return _FALLBACK


def _groq(system: str, user: str, api_key: str) -> str | None:
    try:
        import groq
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = groq.Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        content = resp.choices[0].message.content.strip()
        if not content:
            logger.warning("Groq returned empty content for model %s", model)
            return None
        return content
    except Exception as e:
        logger.error("Groq call failed: %s", e)
        return None


def _anthropic(system: str, user: str, api_key: str) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except Exception as e:
        logger.error("Anthropic call failed: %s", e)
        return None


def _ollama(system: str, user: str, model: str) -> str | None:
    # think:false disables Qwen3 chain-of-thought so content is never empty (Ollama >= 0.7)
    payload = json.dumps({
        "model": model,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"num_predict": 400},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            content = data["message"]["content"].strip()
            if not content:
                logger.warning("Ollama returned empty content for model %s", model)
                return None
            return content
    except Exception as e:
        logger.error("Ollama call failed (model=%s): %s", model, e)
        return None
