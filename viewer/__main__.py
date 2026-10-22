"""CLI entry point: `python -m viewer [--data-dir ...] [--features-dir ...] [--port ...]`."""
import argparse
import os
import sys

from streamlit.web import cli as stcli


def main():
    ap = argparse.ArgumentParser(prog="python -m viewer", description="Browse patents and plot their features in the browser.")
    ap.add_argument("--data-dir", default="data/patent_data_small", help="folder of patents_ipa*.json files")
    ap.add_argument("--features-dir", default="features", help="folder of .csv/.parquet feature files keyed by doc_number")
    ap.add_argument("--sample", help="large-dataset sample (experiment-big-data/sample.py output, e.g. "
                    "data/large/sample_10000.jsonl) instead of --data-dir; embed it first with experiment-big-data/run.py")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    app = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = [
        "streamlit", "run", app,
        "--server.port", str(args.port),
        "--server.headless", str(args.no_browser).lower(),
        "--", "--data-dir", args.data_dir, "--features-dir", args.features_dir,
        *(["--sample", args.sample] if args.sample else []),
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
