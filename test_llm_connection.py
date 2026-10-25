"""Check Ollama Cloud configuration, or make one request with --live."""

import argparse
import sys

from llm_client import OllamaCloudError, chat, validate_configuration

_TEST_QUESTION = "Reply with exactly: Ollama Cloud connection verified."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="send a request to Ollama Cloud (may consume API quota)",
    )
    args = parser.parse_args()

    try:
        host, model = validate_configuration()
    except RuntimeError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if not args.live:
        print(f"Local configuration OK: {host}, model {model}.")
        print("No API request was sent. Use --live only when ready.")
        return 0

    try:
        response = chat([{"role": "user", "content": _TEST_QUESTION}])
    except OllamaCloudError as error:
        print(f"Connection test failed: {error}", file=sys.stderr)
        return 1

    print(response.message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
