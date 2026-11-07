from afl_assistant.chat import AFLChatAgent
from afl_assistant.graph import run_agent
from afl_assistant.memory import ConversationMemory


def test_greeting_intent():
    result = run_agent("Assalam-o-Alaikum")
    assert result["intent"] == "greeting"
    assert result["response"]


def test_off_topic_intent_is_refused():
    result = run_agent("What's the weather today?")
    assert result["intent"] == "off_topic"
    assert "AFL" in result["response"]


def test_team_record_intent_answers_pool_of_teams():
    result = run_agent("Collingwood Magpies ka record kya hai?")
    assert result["intent"] == "team_record"
    assert "wins" in result["response"].lower() or "records" in result["response"].lower()


def test_prediction_requires_two_teams():
    result = run_agent("Predict the Collingwood match")
    assert result["needs_clarification"] is True


def test_prediction_with_two_teams_includes_disclaimer():
    result = run_agent("Who will win Collingwood Magpies vs Carlton Blues?")
    assert result["intent"] == "prediction"
    assert "certainty" in result["response"].lower()


def test_chat_agent_memory_is_bounded():
    memory = ConversationMemory(max_turns=4)
    agent = AFLChatAgent(memory=memory)
    for _ in range(4):
        agent.chat("hello")
    assert len(memory.history) <= 4
