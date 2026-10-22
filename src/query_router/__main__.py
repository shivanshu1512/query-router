"""
Main entry point for the query_router package.

Usage:
    python -m query_router --help
    python -m query_router smoke      # Verify API keys
    python -m query_router oracle     # Run oracle pipeline
    python -m query_router train      # Train router
    python -m query_router serve      # Start API server
    python -m query_router analyze    # Analyze results
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "smoke":
        from query_router.smoke_test import main as smoke_main
        smoke_main()

    elif cmd == "oracle":
        # Forward remaining args to oracle
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        import argparse
        parser = argparse.ArgumentParser(description="Run the oracle pipeline")
        parser.add_argument("--large-model", default="gpt-4o")
        parser.add_argument("--small-model", default="gpt-4o-mini")
        parser.add_argument("--judge-model", default="gpt-4o")
        parser.add_argument("--max-queries", type=int, default=None)
        parser.add_argument("--resume-from", default=None)
        parser.add_argument("--output-dir", default="data/results")
        args = parser.parse_args()
        from query_router.oracle import run_oracle_pipeline
        run_oracle_pipeline(
            large_model=args.large_model,
            small_model=args.small_model,
            judge_model=args.judge_model,
            output_dir=args.output_dir,
            max_queries=args.max_queries,
            resume_from=args.resume_from,
        )

    elif cmd == "train":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        import argparse
        parser = argparse.ArgumentParser(description="Train the query router")
        parser.add_argument("--oracle-results", required=True)
        parser.add_argument("--save-path", default="models/router.pkl")
        args = parser.parse_args()
        from query_router.router import train_router_from_file
        train_router_from_file(args.oracle_results, args.save_path)

    elif cmd == "serve":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "query_router.server:app",
            "--reload", "--port", "8000",
        ])

    elif cmd == "analyze":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        import argparse
        parser = argparse.ArgumentParser(description="Analyze oracle results")
        parser.add_argument("--results", required=True)
        args = parser.parse_args()
        from query_router.analyze import analyze_results
        analyze_results(args.results)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
