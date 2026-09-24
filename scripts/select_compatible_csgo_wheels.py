"""Print locally cached pinned wheels compatible with this Python/platform."""

from pathlib import Path
import sys

# pip is already bootstrapped by setup_csgo_seen10.sh. Its vendored packaging
# understands both CPython and platform wheel tags used by pip itself.
from pip._vendor.packaging import tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import Version


def compatible_wheels(wheel_dir: Path, pins: list[str]) -> list[Path]:
    supported = {tag: rank for rank, tag in enumerate(tags.sys_tags())}
    wheels = []
    for path in wheel_dir.glob("*.whl"):
        try:
            name, version, _, wheel_tags = parse_wheel_filename(path.name)
        except ValueError:
            continue
        compatible = wheel_tags & supported.keys()
        if compatible:
            wheels.append((name, version, min(supported[tag] for tag in compatible), path))

    selected = []
    for pin in pins:
        name, version = pin.split("==", 1)
        matches = [
            wheel for wheel in wheels
            if wheel[0] == canonicalize_name(name) and wheel[1] == Version(version)
        ]
        if matches:
            selected.append(min(matches, key=lambda wheel: wheel[2])[3])
    return selected


if __name__ == "__main__":
    for wheel in compatible_wheels(Path(sys.argv[1]), sys.argv[2:]):
        print(wheel)
