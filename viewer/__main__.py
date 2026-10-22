"""CLI entry point: `python -m viewer [--port ...] [--no-browser]` (or: make viewer)."""
import argparse
import os
import sys

from streamlit.web import cli as stcli


def main():
    ap = argparse.ArgumentParser(prog="python -m viewer", description="Search, browse and inspect the patents in the browser.")
    ap.add_argument("--data-dir", default="data/patent_data_small", help="folder of patents_ipa*.json files")
    ap.add_argument("--features-dir", default="features", help="folder of .csv/.parquet feature files keyed by doc_number")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    app = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = [
        "streamlit", "run", app,
        "--server.port", str(args.port),
        "--server.headless", str(args.no_browser).lower(),
        "--theme.primaryColor", "#2a78d6",
        "--client.toolbarMode", "minimal",
        "--", "--data-dir", args.data_dir, "--features-dir", args.features_dir,
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
