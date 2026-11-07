"""Command-line interface for the AFL assistant."""
from __future__ import annotations

import argparse
import json

from .graph import run_agent
from .predict import predict_match_winner, predict_top_player


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="afl-assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Ask the assistant a question")
    chat.add_argument("message")

    match = sub.add_parser("predict-match", help="Predict a match winner")
    match.add_argument("team_a")
    match.add_argument("team_b")

    player = sub.add_parser("predict-player", help="Rank top players")
    player.add_argument("--team", default=None)
    player.add_argument("--stat", default="disposals", choices=["disposals", "goals", "fantasy_points"])
    player.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    if args.command == "chat":
        print(json.dumps(run_agent(args.message), indent=2, ensure_ascii=False))
    elif args.command == "predict-match":
        print(json.dumps(predict_match_winner(args.team_a, args.team_b), indent=2))
    else:
        print(
            json.dumps(
                predict_top_player(team=args.team, stat_type=args.stat, top_k=args.top_k),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
