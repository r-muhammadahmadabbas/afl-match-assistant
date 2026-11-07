import pytest

from afl_assistant.predict import (
    STAT_COLUMNS,
    build_match_features,
    predict_match_winner,
    predict_top_player,
)


def test_build_match_features_has_all_model_columns():
    from afl_assistant.config import MATCH_FEATURES

    features = build_match_features("Collingwood Magpies", "Carlton Blues")
    assert set(features) == set(MATCH_FEATURES)


def test_predict_match_winner_returns_valid_probability():
    result = predict_match_winner("Collingwood Magpies", "Carlton Blues")
    assert result["winner"] in {
        "Collingwood Magpies",
        "Carlton Blues",
    }
    assert 0.0 <= result["probability"] <= 1.0
    assert result["confidence"] in {"low", "medium", "high"}


def test_predict_top_player_respects_top_k():
    players = predict_top_player(
        team="Collingwood Magpies", stat_type="disposals", top_k=3
    )
    assert len(players) <= 3
    for player in players:
        assert player["stat"] == "disposals"
        assert player["player"]


def test_predict_top_player_rejects_unknown_stat():
    with pytest.raises(ValueError):
        predict_top_player(stat_type="hitouts")


def test_stat_columns_cover_all_supported_stats():
    assert set(STAT_COLUMNS) == {"disposals", "goals", "fantasy_points"}
