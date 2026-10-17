from doosra.analyzer import analyze
from doosra.explain import explain, extract_flags, numbers_are_grounded, facts_from_analysis
from doosra.markets import BookQuote

QUOTES = [BookQuote("pinnacle", (1.78, 2.14)), BookQuote("bet365", (1.72, 2.10))]
MATCH = {"id": "m", "league": "Test League"}


class Fake:
    def __init__(self, text=None, exc=None):
        self.text, self.exc = text, exc

    def complete(self, system, prompt, max_tokens):
        if self.exc:
            raise self.exc
        return self.text


def _analysis(probs=(0.60, 0.40)):
    return analyze(["A", "B"], QUOTES, model_probs=list(probs), bankroll=1000)


def test_grounded_text_is_used():
    an = _analysis()
    f = facts_from_analysis(MATCH, an)
    a = f["sides"][0]
    text = f"A is best at {a['best_odds']:.2f} with a fair chance near {a['fair_chance_pct']}%."
    out = explain(MATCH, an, client=Fake(text))
    assert out["source"] == "llm" and out["text"] == text


def test_invented_number_falls_back_to_template():
    an = _analysis()
    out = explain(MATCH, an, client=Fake("A wins 73% of night games and is a lock."))
    assert out["source"] == "template" and "73" in out["reason"]
    ok, bad = numbers_are_grounded("A wins 73% of night games", facts_from_analysis(MATCH, an))
    assert not ok and bad == ["73"]


def test_errors_and_missing_key_fall_back():
    an = _analysis()
    assert explain(MATCH, an, client=Fake(exc=RuntimeError("boom")))["source"] == "template"
    out = explain(MATCH, an, client=None)  # no ANTHROPIC_API_KEY in test env
    assert out["source"] == "template" and "A is best priced" in out["text"]


def test_template_handles_no_model():
    out = explain(MATCH, analyze(["A", "B"], QUOTES), client=None)
    assert "No verdict" in out["text"]


def test_flags_keep_only_vocab_with_real_evidence():
    notes = "India confirm their XI. Bumrah is ruled out with a back injury. Heavy dew expected tonight."
    raw = ('{"flags": ['
           '{"code": "key_bowler_out", "team": "India", "evidence": "Bumrah is ruled out"},'
           '{"code": "dew_expected", "team": null, "evidence": "Heavy dew expected"},'
           '{"code": "made_up_code", "team": "India", "evidence": "India confirm"},'
           '{"code": "rain_risk", "team": null, "evidence": "storms all day"}]}')
    out = extract_flags(notes, ["India", "Australia"], client=Fake(raw))
    codes = [f["code"] for f in out["flags"]]
    assert codes == ["key_bowler_out", "dew_expected"]
    assert "do not change any probability" in out["note"]


def test_flags_bad_json_and_empty_notes():
    assert extract_flags("", ["A"], client=Fake("{}"))["flags"] == []
    assert extract_flags("some notes", ["A"], client=Fake("not json"))["flags"] == []
