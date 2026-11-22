#!/usr/bin/env python3
"""Version and release notes of the library, for the release workflow.

    python tools/release_info.py version            print the version defined in include/namturbo/version.h, for example 1.0.0-rc.1
    python tools/release_info.py check vTAG         exit with an error if the tag is not "v" + the version
    python tools/release_info.py notes vTAG         print the section of CHANGELOG.md for the version of the tag
    python tools/release_info.py prerelease vTAG    print "true" if the tag has a pre-release identifier, otherwise "false"
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def version() -> str:
    text = (ROOT / "include" / "namturbo" / "version.h").read_text(encoding="utf-8")
    numbers = re.search(r"kVersion = \{(\d+), (\d+), (\d+)\}", text)
    suffix = re.search(r'kPreRelease = "([^"]*)"', text)
    if not numbers or not suffix:
        raise SystemExit("cannot read the version from include/namturbo/version.h")
    base = ".".join(numbers.groups())
    return f"{base}-{suffix.group(1)}" if suffix.group(1) else base


def tag_version(tag: str) -> str:
    if not tag.startswith("v"):
        raise SystemExit(f"the tag {tag!r} does not start with 'v'")
    return tag[1:]


def notes(ver: str) -> str:
    lines = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(rf"## \[{re.escape(ver)}\]", l)), None)
    if start is None:
        raise SystemExit(f"CHANGELOG.md has no section for version {ver}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    return "\n".join(lines[start + 1 : end]).strip() + "\n"


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "version":
        print(version())
    elif cmd == "check":
        tag = tag_version(argv[2])
        if tag != version():
            print(f"the tag says {tag} but version.h defines {version()}", file=sys.stderr)
            return 1
    elif cmd == "notes":
        print(notes(tag_version(argv[2])), end="")
    elif cmd == "prerelease":
        print("true" if "-" in tag_version(argv[2]) else "false")
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
