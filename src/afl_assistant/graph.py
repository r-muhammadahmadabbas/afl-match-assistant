"""LangGraph orchestration for the AFL assistant."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .chat import AFLChatAgent
from .data import resolve_teams_in_text
from .guardrails import is_goodbye, is_greeting, is_off_topic, refusal_response
from .predict import predict_match_winner


class AFLState(TypedDict, total=False):
    user_query: str
    conversation_history: List[Dict[str, str]]
    detected_intent: Optional[str]
    tool_results: Optional[Dict]
    final_response: str
    needs_clarification: bool
    prediction_disclaimer: Optional[str]


def router_node(state: AFLState) -> AFLState:
    query = state["user_query"].lower()
    if is_off_topic(query):
        intent = "off_topic"
    elif is_greeting(query):
        intent = "greeting"
    elif is_goodbye(query):
        intent = "goodbye"
    elif any(k in query for k in ("predict", "will win", "who will", "forecast", "odds")):
        intent = "prediction"
    elif any(k in query for k in ("head to head", "h2h", "matchup")):
        intent = "head_to_head"
    elif any(k in query for k in ("player", "disposals", "goals", "fantasy", "stats")):
        intent = "player_stats"
    elif any(k in query for k in ("record", "win", "loss", "season")):
        intent = "team_record"
    else:
        intent = "general"
    return {**state, "detected_intent": intent}


def retrieval_node(state: AFLState) -> AFLState:
    agent = AFLChatAgent()
    result = agent.chat(state["user_query"])
    return {**state, "final_response": result["response"]}


def prediction_node(state: AFLState) -> AFLState:
    teams = resolve_teams_in_text(state["user_query"])
    if len(teams) < 2:
        return {
            **state,
            "final_response": "Prediction ke liye do teams ke naam batayein.",
            "needs_clarification": True,
            "tool_results": None,
        }

    result = predict_match_winner(teams[0], teams[1])
    return {
        **state,
        "final_response": (
            f"Prediction: {result['winner']} has a {result['probability']:.0%} "
            f"win probability (confidence: {result['confidence']})."
        ),
        "prediction_disclaimer": "Based on historical data, not a certainty.",
        "tool_results": {"prediction": result},
    }


def refusal_node(state: AFLState) -> AFLState:
    return {**state, "final_response": refusal_response()}


def validation_node(state: AFLState) -> AFLState:
    if state.get("detected_intent") == "prediction" and not state.get("tool_results"):
        return {**state, "needs_clarification": True}
    return state


def response_node(state: AFLState) -> AFLState:
    response = state.get("final_response", "")
    if state.get("prediction_disclaimer") and "certainty" not in response:
        response += f"\n\n{state['prediction_disclaimer']}"
    return {**state, "final_response": response}


def route_intent(state: AFLState) -> str:
    return {
        "prediction": "prediction_node",
        "off_topic": "refusal_node",
    }.get(state.get("detected_intent", ""), "retrieval_node")


def build_graph():
    workflow = StateGraph(AFLState)
    workflow.add_node("router_node", router_node)
    workflow.add_node("prediction_node", prediction_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("refusal_node", refusal_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("response_node", response_node)

    workflow.set_entry_point("router_node")
    workflow.add_conditional_edges("router_node", route_intent)
    for node in ("prediction_node", "retrieval_node", "refusal_node"):
        workflow.add_edge(node, "validation_node")
    workflow.add_edge("validation_node", "response_node")
    workflow.add_edge("response_node", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def run_agent(query: str, history: Optional[List[Dict]] = None) -> Dict:
    state: AFLState = {
        "user_query": query,
        "conversation_history": history or [],
        "detected_intent": None,
        "tool_results": None,
        "final_response": "",
        "needs_clarification": False,
        "prediction_disclaimer": None,
    }
    result = get_graph().invoke(state)
    return {
        "response": result.get("final_response", ""),
        "intent": result.get("detected_intent", ""),
        "needs_clarification": result.get("needs_clarification", False),
    }
