import json

import httpx
import pytest

from doosra.llm import GEMINI_URL, GROQ_URL, Chain, GeminiClient, GroqClient, LLMError, from_env


def gemini_ok(text="Gemini says hi"):
    def h(req):
        assert req.headers["x-goog-api-key"] == "g-key"
        body = json.loads(req.content)
        assert body["system_instruction"]["parts"][0]["text"] == "sys"
        assert body["contents"][0]["parts"][0]["text"] == "prompt"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})
    return httpx.MockTransport(h)


def groq_ok(text="Groq says hi"):
    def h(req):
        assert str(req.url) == GROQ_URL and req.headers["authorization"] == "Bearer q-key"
        body = json.loads(req.content)
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})
    return httpx.MockTransport(h)


def fails(status=500):
    return httpx.MockTransport(lambda req: httpx.Response(status, json={"error": "nope"}))


def test_gemini_request_and_parse():
    c = GeminiClient("g-key", model="m1", transport=gemini_ok())
    assert c.complete("sys", "prompt", 50) == "Gemini says hi"
    assert GEMINI_URL.format(model="m1").endswith("/models/m1:generateContent")


def test_groq_request_and_parse():
    assert GroqClient("q-key", transport=groq_ok()).complete("sys", "prompt", 50) == "Groq says hi"


@pytest.mark.parametrize("client", [GeminiClient("g", transport=fails(429)), GroqClient("q", transport=fails(401))])
def test_provider_errors_raise(client):
    with pytest.raises(LLMError):
        client.complete("s", "p", 10)


def test_gemini_empty_candidates_is_an_error():
    c = GeminiClient("g", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"candidates": []})))
    with pytest.raises(LLMError):
        c.complete("s", "p", 10)


def test_chain_falls_back_and_remembers_who_answered():
    chain = Chain([GeminiClient("g", transport=fails(429)), GroqClient("q-key", transport=groq_ok())])
    assert chain.complete("sys", "prompt", 50) == "Groq says hi"
    assert chain.last_used == "groq"


def test_chain_all_failing_raises_with_reasons():
    chain = Chain([GeminiClient("g", transport=fails()), GroqClient("q", transport=fails())])
    with pytest.raises(LLMError, match="gemini.*groq"):
        chain.complete("s", "p", 10)


def test_from_env_order_and_absence():
    assert from_env({}) is None
    c = from_env({"GROQ_API_KEY": "q", "GEMINI_API_KEY": "g"})
    assert [x.name for x in c.clients] == ["gemini", "groq"]
    c = from_env({"GROQ_API_KEY": "q", "GEMINI_API_KEY": "g", "DOOSRA_LLM_ORDER": "groq,gemini"})
    assert [x.name for x in c.clients] == ["groq", "gemini"]
    assert [x.name for x in from_env({"ANTHROPIC_API_KEY": "a"}).clients] == ["claude"]


def test_explain_uses_the_chain_and_still_guards_numbers():
    from doosra.analyzer import analyze
    from doosra.explain import explain
    from doosra.markets import BookQuote
    an = analyze(["A", "B"], [BookQuote("pinnacle", (1.8, 2.1))], [0.6, 0.4])
    good = Chain([GroqClient("q-key", transport=groq_ok("A is best priced at 1.80."))])
    out = explain({"league": "L"}, an, client=good)
    assert out["source"] == "llm" and out["provider"] == "groq"
    bad = Chain([GroqClient("q-key", transport=groq_ok("A wins 91% of the time."))])
    assert explain({"league": "L"}, an, client=bad)["source"] == "template"
