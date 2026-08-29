"""
``python -m hdr_platform <subcommand>``

A single entry point for the two things a job script asks about:

    python -m hdr_platform site      [--json] [--field NAME] [--site NAME]
    python -m hdr_platform doctor    [--json] [--strict] [--skip ...]

Both subcommands are also reachable as ``python -m hdr_platform.site`` and
``python -m hdr_platform.doctor``; this form exists so the shell helpers
have one spelling to remember.
"""

import sys

_USAGE = __doc__.strip()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0

    sub, rest = argv[0], argv[1:]
    if sub == "site":
        from .site import _main
        return _main(rest)
    if sub == "doctor":
        from .doctor import main as doctor_main
        return doctor_main(rest)

    print(f"unknown subcommand {sub!r}\n\n{_USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
