"""YAML CLI defaults, with config-relative paths and explicit CLI precedence."""
import argparse
from pathlib import Path
import sys


def parse_config(parser, *, required=(), source_required=False, argv=None):
    parser.add_argument("--config", type=Path, help="YAML defaults; CLI arguments take precedence")
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    probe.add_argument("--project-root", type=Path)
    overrides = probe.parse_known_args(argv)[0]
    config_path = overrides.config
    if config_path is not None:
        try:
            import yaml
            document = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("Config must be a YAML mapping")
            actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}
            unknown = document.keys() - actions.keys()
            if unknown:
                raise ValueError(f"Unknown config fields: {sorted(unknown)}")
            base = config_path.resolve().parent
            # project_root in YAML is relative to the config, all other paths to that root.
            if overrides.project_root is not None:
                base = overrides.project_root.resolve()
                document["project_root"] = str(base)
            elif document.get("project_root") is not None:
                base = (base / document["project_root"]).resolve()
                document["project_root"] = str(base)
            defaults = {}
            for name, value in document.items():
                action = actions[name]
                if value is None:
                    defaults[name] = None
                elif action.type is Path:
                    if not isinstance(value, str):
                        raise ValueError(f"{name} must be a path string")
                    defaults[name] = (base / value).resolve()
                elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
                    if not isinstance(value, bool):
                        raise ValueError(f"{name} must be a boolean")
                    defaults[name] = value
                elif action.type is int:
                    if type(value) is not int:
                        raise ValueError(f"{name} must be an integer")
                    defaults[name] = value
                else:
                    defaults[name] = action.type(value) if action.type else value
            # Switching the source on the CLI replaces the YAML source mode.
            if any(a == "--input-dir" or a.startswith("--input-dir=") for a in argv):
                defaults["metadata"] = None
            if any(a == "--metadata" or a.startswith("--metadata=") for a in argv):
                defaults["input_dir"] = None
            parser.set_defaults(**defaults)
        except (ImportError, OSError, ValueError, TypeError) as exc:
            parser.error(f"Cannot load config: {exc}")
        except yaml.YAMLError as exc:
            parser.error(f"Invalid YAML: {exc}")
    args = parser.parse_args(argv)
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        parser.error("Required arguments/config fields: " + ", ".join(missing))
    if source_required and (args.input_dir is None) == (args.metadata is None):
        parser.error("Choose exactly one of --input-dir or --metadata")
    return args
