from afl_assistant.guardrails import (
    is_goodbye,
    is_greeting,
    is_off_topic,
    is_prediction_query,
    refusal_response,
)


def test_off_topic_queries_are_refused():
    assert is_off_topic("What's the weather in Lahore?")
    assert is_off_topic("Tell me about the stock market")


def test_afl_queries_are_not_off_topic():
    assert not is_off_topic("Collingwood ka record kya hai?")
    assert not is_off_topic("Who has the most disposals?")


def test_greeting_and_goodbye_detection():
    assert is_greeting("Assalam-o-Alaikum")
    assert is_greeting("hello")
    assert is_goodbye("bye")
    assert is_goodbye("Khuda Hafiz")


def test_prediction_query_detection():
    assert is_prediction_query("Who will win Collingwood vs Carlton?")
    assert is_prediction_query("predict the match")
    assert not is_prediction_query("Collingwood record")


def test_refusal_response_is_afl_redirect():
    text = refusal_response()
    assert "AFL" in text
