# Eval data

* `retrieval_corpus.jsonl` / `retrieval_queries.jsonl` — a **small hand-written, synthetic**
  labelled set (28 short news-style docs, 16 queries with relevant doc ids) used for the
  offline retrieval eval (recall@k, MRR). It is a smoke test for the retrieval stack, not a
  benchmark; swap in your own labelled data for real measurements.
* `forecast_snapshot.json` — produced by `pmra eval snapshot` from the **live** Polymarket
  Gamma/CLOB APIs and news sources (titles, links, dates only), so the forecast eval can be
  replayed offline in CI. Regenerate it any time.
