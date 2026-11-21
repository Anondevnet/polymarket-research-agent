**Markets:** 8 (PARTIAL: 8/30 scored) · **horizon:** 7d before close · **LLM:** `gemini-2.5-flash-lite` · **embedder:** `hashing-512` · **base rate YES:** 0.375

| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---|---|---|
| agent (gemini-2.5-flash-lite) | 0.1941 | 0.5988 | 0.3538 |
| market price @ as_of | 0.1313 | 0.3991 | 0.1583 |
| baseline 0.5 | 0.25 | 0.6931 | 0.125 |

**Subset forecast after LLM knowledge cutoff (2025-01-31):** 7 markets · base rate YES: 0.2857

| Forecaster | Brier ↓ | Log loss ↓ | ECE ↓ |
|---|---|---|---|
| agent (gemini-2.5-flash-lite) | 0.2215 | 0.677 | 0.3971 |
| market price @ as_of | 0.1501 | 0.4551 | 0.1799 |
| baseline 0.5 | 0.25 | 0.6931 | 0.2143 |
