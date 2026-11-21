**Markets:** 30 · **horizon:** 7d before close · **LLM:** `heuristic-lexicon` · **embedder:** `hashing-512` · **base rate YES:** 0.3

| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---|---|---|
| agent (heuristic-lexicon) | 0.2856 | 0.7713 | 0.2657 |
| market price @ as_of | 0.1781 | 0.5581 | 0.176 |
| baseline 0.5 | 0.25 | 0.6931 | 0.2 |

**Subset forecast after LLM knowledge cutoff (2025-01-31):** 28 markets · base rate YES: 0.2857

| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---|---|---|
| agent (heuristic-lexicon) | 0.2913 | 0.7841 | 0.273 |
| market price @ as_of | 0.1908 | 0.5977 | 0.1883 |
| baseline 0.5 | 0.25 | 0.6931 | 0.2143 |
