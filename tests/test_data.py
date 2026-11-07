from afl_assistant.data import (
    load_feature_table,
    load_home_matches,
    load_player_game,
    normalize_team_name,
    resolve_teams_in_text,
)


def test_normalize_team_name_resolves_aliases():
    assert normalize_team_name("pies") == "Collingwood Magpies"
    assert normalize_team_name("Blues") == "Carlton Blues"
    assert normalize_team_name("collingwood magpies") == "Collingwood Magpies"


def test_normalize_team_name_returns_input_when_unknown():
    assert normalize_team_name("Nowhere United") == "Nowhere United"


def test_resolve_teams_in_text_finds_two_teams():
    teams = resolve_teams_in_text("Collingwood vs Carlton prediction")
    assert "Collingwood Magpies" in teams
    assert "Carlton Blues" in teams


def test_processed_datasets_have_expected_columns():
    features = load_feature_table()
    expected = {
        "win_streak",
        "avg_score_3",
        "h2h_win_rate",
        "days_rest",
        "home_win",
    }
    assert expected.issubset(features.columns)
    assert len(features) > 0


def test_home_matches_are_chronologically_sorted():
    matches = load_home_matches()
    assert matches["match_date"].is_monotonic_increasing


def test_player_game_dataset_contains_target_columns():
    games = load_player_game()
    assert {"player_id", "team", "disposals_target", "goals_target"}.issubset(
        games.columns
    )
    assert games["player_id"].notna().all()
