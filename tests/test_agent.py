import json

from conftest import DOCS, MARKET, FakeOpenAI, dt
from pm_research.llm import HeuristicLLM, OpenAIChatLLM
from pm_research.service import ResearchService

AS_OF = dt("2025-09-10T00:00:00")


def test_graph_end_to_end_heuristic(service: ResearchService) -> None:
    fc = service.research_market("m1", as_of=AS_OF)
    assert 0 < fc.probability < 1
    assert fc.market_price == 0.62
    assert fc.edge == round(fc.probability - 0.62, 4)
    assert service.store.count("m1") > 0  # gather node ingested evidence
    # look-ahead guard: post-cutoff article and off-topic article not cited
    titles = " ".join(fc.evidence.evidence_for + fc.evidence.evidence_against)
    assert "Fed cuts rates by a quarter point" not in titles


def test_search_evidence_lazily_ingests(service: ResearchService) -> None:
    hits = service.search_evidence("m1", query="Powell rate cut", k=3)
    assert hits[0].chunk.doc_id == "d1"


def test_heuristic_stance() -> None:
    assert HeuristicLLM.stance("Candidate wins and leads polls") > 0
    assert HeuristicLLM.stance("Bill rejected, vote fails") < 0


def test_graph_with_openai_compatible_llm(service: ResearchService) -> None:
    fake = FakeOpenAI(
        [
            json.dumps(
                {
                    "evidence_for": ["Powell signals cut [1]"],
                    "evidence_against": ["Hot CPI [2]"],
                    "summary": "Mixed but leaning yes.",
                }
            ),
            json.dumps({"probability": 0.71, "rationale": "Chair guidance outweighs CPI."}),
        ]
    )
    llm = OpenAIChatLLM("test-model", client=fake)
    svc = ResearchService(
        service.settings,
        polymarket=service.polymarket,
        sources=service.sources,
        embedder=service.embedder,
        store=service.store,
        llm=llm,
    )
    fc = svc.research(MARKET, as_of=AS_OF)
    assert fc.probability == 0.71
    assert fc.evidence.evidence_for == ["Powell signals cut [1]"]
    # the market price is deliberately hidden from the estimate prompt
    assert "0.62" not in fake.calls[1]["messages"][1]["content"]
    # post-cutoff doc never reaches the LLM
    assert "quarter point" not in fake.calls[0]["messages"][1]["content"]
    assert len(DOCS) == 4


def test_unparseable_llm_output_falls_back() -> None:
    fake = FakeOpenAI(["not json", '{"probability": "oops"}'])
    llm = OpenAIChatLLM("m", client=fake)
    ev = llm.summarize(MARKET, [])
    p, _ = llm.estimate(MARKET, ev, None)
    assert ev.evidence_for == [] and p == 0.5
