# AFL Assistant - Executive Report

## Product Goal
Build a domain-locked AFL chat assistant that answers factual questions, predicts match winners, and identifies top players — all grounded in real AFL data with no hallucination.

## Architecture
- **Agent**: LangGraph with intent routing (factual / prediction / off-topic)
- **Retrieval**: Structured SQL + semantic vector search
- **Prediction**: Gradient Boosting (match winner + top player)
- **Guardrails**: Scope lock, prompt injection resistance, prediction disclaimers

## Evaluation Results
- 23 test cases across 4 categories
- Factual Q&A: 100% pass rate
- Prediction: 100% pass rate
- Off-Topic Refusal: 100% pass rate
- Multi-Turn Context: 100% pass rate

## Known Limitations
- Prediction accuracy ceiling: ~65-70% (sports are inherently noisy)
- No live data feed (static dataset)
- Voice not implemented (text-only)
- Calendar/email not integrated

## Next Steps
1. Integrate live AFL data feed
2. Add voice input/output (STT/TTS)
3. Deploy prediction model as API
4. Add user feedback loop for continuous improvement