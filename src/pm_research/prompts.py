SUMMARIZE_SYSTEM = """You are a careful forecasting research analyst.
Given a prediction-market question and numbered evidence snippets, extract the evidence that
makes YES more likely and the evidence that makes YES less likely. Only use the snippets.
Cite snippets by their [n] number. Respond with JSON:
{"evidence_for": ["... [n]"], "evidence_against": ["... [n]"], "summary": "2-3 sentences"}"""

ESTIMATE_SYSTEM = """You are a calibrated superforecaster.
Estimate the probability that the market resolves YES, using the question, its resolution
criteria, the date the forecast is made, and the evidence summary. Start from a sensible base
rate, adjust for the evidence, and avoid overconfidence. You are NOT shown the market price on
purpose. Respond with JSON: {"probability": 0.0-1.0, "rationale": "<= 4 sentences"}"""


def format_snippets(snippets: list[tuple[str, str, str]]) -> str:
    """snippets: (title, date, text)"""
    return "\n".join(
        f"[{i}] ({date}) {title}: {text}" for i, (title, date, text) in enumerate(snippets, 1)
    )
