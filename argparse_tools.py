import argparse
from pathlib import Path

import yaml
import sys


class ArgumentParser(argparse.ArgumentParser):
    """Simple implementation of ArgumentParser supporting config file

    This class is originated from https://github.com/bw2/ConfigArgParse,
    but this class is lack of some features that it has.

    - Not supporting multiple config files
    - Automatically adding "--config" as an option.
    - Not supporting any formats other than yaml
    - Not checking argument type

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", help="Give config file in yaml format")

    def parse_known_args(self, args=None, namespace=None):
        # Once parsing for setting from "--config"
        _args, _ = super().parse_known_args(args, namespace)
        if _args.config is not None:
            if not Path(_args.config).exists():
                self.error(f"No such file: {_args.config}")

            with open(_args.config, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
            if not isinstance(d, dict):
                self.error("Config file has non dict value: {_args.config}")

            for key in d:
                for action in self._actions:
                    if key == action.dest:
                        break
                else:
                    self.error(f"unrecognized arguments: {key} (from {_args.config})")

            # NOTE(kamo): Ignore "--config" from a config file
            # NOTE(kamo): Unlike "configargparse", this module doesn't check type.
            #   i.e. We can set any type value regardless of argument type.
            self.set_defaults(**d)
        return super().parse_known_args(args, namespace)


def get_commandline_args():
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''")
        if all(char not in arg for char in extra_chars)
        else "'" + arg.replace("'", "'\\''") + "'"
        for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)