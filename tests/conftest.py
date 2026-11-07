"""Shared pytest fixtures for the AFL assistant test-suite."""
import pytest

from afl_assistant.data import load_home_matches


@pytest.fixture(scope="session")
def home_matches():
    return load_home_matches()


@pytest.fixture(scope="session")
def known_teams(home_matches):
    return sorted(home_matches["team_name"].dropna().unique().tolist())
