"""AFL match-winner and top-player assistant."""

from .graph import run_agent
from .predict import predict_match_winner, predict_top_player

__all__ = ["run_agent", "predict_match_winner", "predict_top_player"]
__version__ = "1.0.0"
