# AFL Match Assistant

[![CI](https://github.com/your-org/afl-match-assistant/actions/workflows/ci.yml/badge.svg)](../../actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Model Results](https://img.shields.io/badge/accuracy-0.6204-brightgreen)](README.md#model-results)

A domain‑locked AFL assistant that answers factual questions, predicts match
winners, and ranks the most productive players — grounded in real historical
data, with hard guardrails that keep it from wandering off‑topic.

The service combines three layers:

1. **Structured retrieval** over 40+ seasons of match and player data.
2. **Machine‑learning prediction** for match outcomes and player output.
3. **A LangGraph agent** that routes each question to the right layer and adds
   prediction disclaimers.

Everything runs locally. No paid APIs are required.

---

## Highlights

| Capability | Implementation |
| --- | --- |
| Factual Q&A | Team records, head‑to‑head history, player career stats |
| Match prediction | Gradient‑boosting classifier on leak‑free rolling form |
| Top‑player ranking | One regressor per stat (disposals, goals, fantasy) |
| Domain lock | Keyword guardrails + safe refusal response |
| Multi‑turn memory | Bounded conversation transcript |
| API | FastAPI with `/chat`, `/predict/match`, `/predict/player` |
| Ops | Docker, Compose, Makefile, GitHub Actions |

---

## Model Results

Models are trained on every completed match up to the penultimate season, and
evaluated on the most recent season. The split is strictly chronological.

### Match Winner

Trained on 14,944 team‑match rows (1983–2023), validated on the 2024 season
(432 rows), tested on 2025 (432 rows).

| Model | Val accuracy | Val ROC AUC | Val Brier |
| --- | --- | --- | --- |
| Logistic regression | 0.5880 | 0.6297 | 0.2373 |
| **Gradient boosting (selected)** | **0.5949** | **0.6410** | **0.2341** |

Hold‑out season (2025):

| Metric | Value |
| --- | --- |
| Accuracy | **0.6204** |
| F1 | 0.6114 |
| ROC AUC | 0.6760 |
| Brier score | 0.2263 |

For context, the majority‑class baseline on this season is ~0.56. The model
beats it by roughly six accuracy points and produces materially better‑ranked
probabilities than a coin flip.

### Player Output Regressors

Each model predicts a player's **next‑game** stat from their previous five
games (rolling averages shifted by one match). Trained on 107,073 player‑games,
tested on the 2025 season (2,808 player‑games).

| Stat | MAE | RMSE |
| --- | --- | --- |
| Disposals | **3.58** | 4.62 |
| Goals | **0.71** | 0.96 |
| Fantasy points | **16.73** | 20.96 |

### A Note on Data Leakage

An earlier version of the feature table included a `win_streak` column that
already contained the current match result, which produced a suspicious 100%
accuracy. `scripts/prepare_data.py` now computes every feature from prior
matches only:

- rolling form uses `shift(1)`;
- `win_streak` is the streak **entering** the match;
- head‑to‑head history is accumulated in chronological league order.

The honest result is a ~62% match‑winner model — in line with published AFL
baselines, where the sport's inherent variance caps realistic accuracy.

---

## Architecture

```text
                          ┌────────────────┐
   user question ───────►│  router (intent)│
                           └───────┬────────┘
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       ┌───────────┐     ┌────────────┐     ┌────────────┐
       │ retrieval │     │ prediction │     │  refusal   │
       │ (records, │     │ (match /   │     │ (off-topic │
       │  stats)   │     │  players)  │     │  guardrail)│
       └─────┬─────┘     └─────┬──────┘     └─────┬──────┘
             └──────────────────┼──────────────────┘
                              ▼
                       ┌─────────────────┑
                       │ validation +    │
                       │ response writer │
                       └─────────────────┘
                              END
```

| Node | Responsibility |
| --- | --- |
| `router_node` | Classify greeting / factual / prediction / off‑topic / goodbye |
| `retrieval_node` | Delegate to the deterministic chat agent |
| `prediction_node` | Build a feature row and score the match model |
| `refusal_node` | Return the AFL‑only redirect message |
| `validation_node` | Flag predictions that are missing team names |
| `response_node` | Append the "not a certainty" disclaimer |

### Package Layout

```text
src/afl_assistant/
├── config.py        # Paths, feature contracts, team aliases, guardrail terms
├── data.py          # CSV loading, team‑name normalization, text team extraction
├── retrieval.py     # Team record, head‑to‑head, player career stats
├── guardrails.py    # Off‑topic / greeting / goodbye / prediction detection
├── memory.py        # Bounded conversation memory
├── training.py      # Match and player model training
├── predict.py       # Match‑winner and top‑player inference
├── chat.py          # Deterministic retrieval‑grounded agent
├── graph.py         # LangGraph orchestration
├── api.py           # FastAPI service
└── cli.py           # Command‑line interface
```

---

## Quick Start

```bash
git clone https://github.com/your-org/afl-match-assistant.git
cd afl-match-assistant

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
pytest                          # run the full test suite (12/12)
```

### Ask a Question

```bash
afl-assistant chat "Collingwood Magpies ka record kya hai?"
```

```json
{
  "response": "Collingwood Magpies: 306 wins, 201 losses (60% win rate across 507 matches).",
  "intent": "team_record",
  "needs_clarification": false
}
```

```bash
afl-assistant predict-match "pies" "blues"
afl-assistant predict-player --team "Collingwood Magpies" --stat disposals --top-k 5
```

### Python API

```python
from afl_assistant import predict_match_winner, predict_top_player, run_agent

run_agent("Who will win Collingwood Magpies vs Carlton Blues?")
predict_match_winner("Collingwood Magpies", "Carlton Blues")
predict_top_player(team="Carlton Blues", stat_type="goals", top_k=3)
```

---

## HTTP API

```bash
uvicorn afl_assistant.api:app --reload
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `POST` | `/chat` | Ask a natural‑language question |
| `POST` | `/predict/match` | Match‑winner probability |
| `GET` | `/predict/player` | Rank players by predicted stat |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Collingwood vs Carlton prediction"}'
```

Interactive docs: <http://localhost:8000/docs>

---

## Data Pipeline

The runtime loads six processed files from `data/processed/`:

| File | Grain |
| --- | --- |
| `feature_table.csv` | one team per match |
| `train.csv` / `val.csv` / `test.csv` | chronological season splits |
| `home_matches_eda.csv` | retrieval‑friendly home‑team match view |
| `player_game_eda.csv` | one player per game, with prior‑form features |

To rebuild everything from the raw exports:

```bash
# place the raw CSVs in data/raw/ (see data/README.md)
python scripts/prepare_data.py
python -m afl_assistant.training
```

The raw round‑by‑round file is ~39 MB and is **not** committed. The player‑info
lookup table (0.6 MB) is committed because the app needs it at runtime.

---

## Guardrails

| Guard | Behaviour |
| --- | --- |
| Off‑topic keywords (weather, politics, crypto, …) | Refuse and redirect to AFL |
| Prediction without two teams | Ask for clarification |
| Every prediction | Appended "not a certainty" disclaimer |
| Prompt‑injection style input | Treated as ordinary text; never executed |
| Unknown team or player | Safe "not found" message, no crash |

---

## Testing

```bash
pytest -q
ruff check src tests
```

The suite covers data contracts, team‑name resolution, guardrails, the chat
graph, and both prediction paths. It runs fully offline against the committed
artifacts.

---

## Docker

```bash
docker compose up --build
```

The API is exposed on <http://localhost:8000>.

---

## Limitations & Next Steps

- **Static data.** There is no live AFL feed; predictions use the last
  committed season.
- **Text only.** Voice input/output is out of scope for this repository.
- **No betting advice.** Outputs are informational probabilities, not tips.
- **Ceiling on accuracy.** AFL results are noisy; ~60–70% is a realistic
  target. Future work should add player‑level availability, travel, and
  umpire/venue signals.

---

## License

MIT. AFL data is used for analytical and educational purposes.