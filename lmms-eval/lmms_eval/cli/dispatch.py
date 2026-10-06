"""CLI dispatcher for the project-specific lmms-eval subset."""

from __future__ import annotations

import argparse
import sys


_SUBCOMMANDS = {"eval", "tasks", "models", "version"}

_BANNER = """LMMs-Eval (GeoGR subset)

  lmms-eval eval   [--model X --tasks Y]   Run evaluation
  lmms-eval tasks  [list|groups|subtasks]  Browse benchmarks
  lmms-eval models [--aliases]             List model backends
  lmms-eval version                        Version info
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmms-eval",
        description="LMMs-Eval runtime for GeoGR",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand")

    from lmms_eval.cli.models_cmd import add_models_parser
    from lmms_eval.cli.tasks_cmd import add_tasks_parser
    from lmms_eval.cli.version_cmd import add_version_parser

    add_tasks_parser(sub)
    add_models_parser(sub)
    add_version_parser(sub)

    eval_parser = sub.add_parser(
        "eval",
        help="Run model evaluation",
        add_help=False,
    )
    eval_parser.set_defaults(func=None)
    return parser


def _is_legacy_invocation(argv: list[str]) -> bool:
    return bool(argv) and argv[0].startswith("-")


def main() -> None:
    argv = sys.argv[1:]

    if not argv:
        print(_BANNER)
        return

    if argv in (["--help"], ["-h"]):
        _build_parser().print_help()
        return

    if _is_legacy_invocation(argv) or argv[0] == "eval":
        if argv[0] == "eval":
            argv = argv[1:]
        sys.argv = [sys.argv[0], *argv]
        from lmms_eval.__main__ import cli_evaluate

        cli_evaluate()
        return

    if argv[0] not in _SUBCOMMANDS:
        _build_parser().error(f"unknown command: {argv[0]}")

    parser = _build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func") and args.func is not None:
        args.func(args)
        return
    parser.print_help()
