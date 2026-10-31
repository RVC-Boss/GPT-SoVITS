"""Command-line entry point for the single-worker inference server."""

import argparse

import uvicorn

from . import app as service_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT-SoVITS inference API")
    parser.add_argument(
        "-c",
        "--tts_config",
        default=service_app.DEFAULT_CONFIG_PATH,
        help="TTS inference configuration",
    )
    parser.add_argument("-a", "--bind_addr", default="127.0.0.1", help="default: 127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    service_app.state.tts_config = service_app.TTS_Config(
        args.tts_config or service_app.DEFAULT_CONFIG_PATH
    )
    host = args.bind_addr
    if host == "None":
        host = None
    uvicorn.run(app=service_app.APP, host=host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
