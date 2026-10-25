"""Unified entry point for the ZentroMart VoiceOps Project."""

import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps project runner")
    parser.add_argument(
        "mode", 
        choices=["rag", "batch", "chat"], 
        help="Execution mode: 'rag' for RAG evaluation, 'batch' for voice batch processing, 'chat' for interactive text agent"
    )
    parser.add_argument("--args", type=str, help="Optional additional arguments for the selected mode")

    args = parser.parse_args()
    
    # Convert args string to list if provided
    extra_args = args.args.split() if args.args else []

    if args.mode == "rag":
        # Default run_rag.py
        cmd = [sys.executable, "run_rag.py"] + extra_args
    elif args.mode == "batch":
        # Default scripts/run_voice_batch.py
        cmd = [sys.executable, "scripts/run_voice_batch.py"] + extra_args
    elif args.mode == "chat":
        # Default scripts/test_text_agent.py
        cmd = [sys.executable, "scripts/test_text_agent.py"] + extra_args
    else:
        print("Invalid mode.")
        sys.exit(1)

    print(f"Executing {args.mode} mode: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: The script for mode '{args.mode}' was not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
