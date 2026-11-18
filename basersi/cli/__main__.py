import ast
import importlib
import os
import pkgutil
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def describe(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    has_main, help_text = False, ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELP" for t in node.targets):
            try:
                help_text = str(ast.literal_eval(node.value))
            except ValueError:
                help_text = ""
    return has_main, help_text


def commands():
    out = []
    for info in pkgutil.iter_modules([PKG_DIR]):
        if info.name.startswith("_"):
            continue
        has_main, help_text = describe(os.path.join(PKG_DIR, info.name + ".py"))
        if has_main:
            out.append((info.name.replace("_", "-"), help_text))
    return sorted(out)


def usage():
    lines = ["usage: basersi <command> [args]   (or: python -m basersi.cli <command> [args])", "", "commands:"]
    cmds = commands()
    width = max([len(c) for c, _ in cmds] + [8])
    for name, doc in cmds:
        lines.append(f"  {name.ljust(width)}  {doc}")
    if not cmds:
        lines.append("  (none installed)")
    from basersi import config
    lines += ["", "Run `basersi <command> --help` for the options of one command.", "Python 3.12; install with `pip install .[gpu]` (add `sources` to build the rows from the official files).", "", config.EXIT_CODES]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name = argv[0].replace("-", "_")
    if name.startswith("_") or not os.path.exists(os.path.join(PKG_DIR, name + ".py")):
        print(f"unknown command: {argv[0]}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    from basersi import config
    mod = importlib.import_module(f"basersi.cli.{name}")
    return config.run_command(mod.main, argv[1:])


if __name__ == "__main__":
    sys.exit(main())
