# AFL Assistant - Demo Script (5-7 Minutes)

## Slide 1: Title (30 seconds)
- Project: AFL Assistant - Domain-Locked Chat + Prediction Agent
- Tagline: "AI-powered AFL expert that answers questions and predicts outcomes"
- Your name, date

## Slide 2: Problem (1 minute)
- AFL fans want quick answers about teams, players, matches
- Existing tools are either too slow or hallucinate stats
- No domain-locked assistant that only discusses AFL

## Slide 3: Solution (1 minute)
- AI assistant that only discusses AFL topics
- Answers factual questions using RAG (no hallucination)
- Predicts match winners and top players
- Offers polite refusal for off-topic queries

## Slide 4: Architecture (1 minute)
- LangGraph for agent orchestration
- Intent routing: factual / prediction / off-topic
- RAG pipeline with ChromaDB vector search
- Gradient Boosting for predictions

## Slide 5: Live Demo - Factual Question (1.5 minutes)
- Type: "Collingwood ka record kya hai?"
- Show: Agent retrieves exact stats from database
- Show: Response is grounded, no hallucination

## Slide 6: Live Demo - Prediction (1.5 minutes)
- Type: "Who will win Collingwood vs Carlton?"
- Show: Agent predicts with probability
- Show: Disclaimer added ("based on historical data")

## Slide 7: Live Demo - Off-Topic Refusal (1 minute)
- Type: "What's the weather like?"
- Show: Agent politely refuses and redirects to AFL

## Slide 8: Evaluation Results (1 minute)
- 23 test cases across 4 categories
- 100% pass rate on factual, prediction, off-topic, multi-turn
- Prompt injection tests passed

## Slide 9: Q&A (30 seconds)
- Thank you
- Questions?