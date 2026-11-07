# AFL Assistant - Monitoring & Maintenance Plan

## 1. Monitoring Checklist

| Metric | Target | Alert Threshold | Frequency | Action |
|--------|--------|-----------------|-----------|--------|
| API Uptime | 99.9% | < 99.5% | Real-time | Investigate + restart |
| Response Latency | < 2s | > 3s | Real-time | Check LLM + STT |
| Prediction Accuracy | > 65% | < 60% | Weekly | Retrain model |
| Off-Topic Leak Rate | 0% | > 0% | Real-time | Strengthen guardrails |
| Error Rate | < 1% | > 2% | Hourly | Check logs |
| RAG Accuracy | > 90% | < 85% | Weekly | Update knowledge base |

## 2. Weekly Maintenance

| Day | Task | Duration |
|-----|------|----------|
| Monday | Review weekend metrics | 30 min |
| Tuesday | Update player stats | 1 hour |
| Wednesday | Review failed predictions | 1 hour |
| Thursday | Update FAQ knowledge base | 30 min |
| Friday | Backup + security review | 30 min |

## 3. Retraining Triggers

| Trigger | Action |
|---------|--------|
| 100 new matches | Retrain match winner model |
| New season | Full model retrain |
| Accuracy drops 5% | Investigate + retrain |
| New players | Update player database |

## 4. Backup Strategy

| Data | Frequency | Retention |
|------|-----------|-----------|
| Match results | Daily | Forever |
| Model artifacts | On deploy | 10 versions |
| API logs | Daily | 90 days |
| Config/Secrets | On change | Infinite |