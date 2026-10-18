"""
Make a delocated macOS wheel share an OpenMP runtime already in the process.

delocate bundles libomp into the wheel (slugpy/.dylibs/libomp.dylib) and
points every binary at that copy by path (@loader_path/.dylibs/...). Then
a process that already has another libomp loaded -- e.g. conda-forge's
numpy, whose OpenBLAS uses conda's llvm-openmp -- ends up with two OpenMP
runtimes, which libomp refuses at initialization ("OMP: Error #15") or
which crash. This script rewrites the wheel so that every binary loads
libomp as @rpath/libomp.dylib, with an rpath to the bundled copy, and
gives the bundled copy the install name @rpath/libomp.dylib too (as
conda-forge's has). dyld reuses an image already loaded under the install
name a binary asks for, so whichever libomp is loaded first -- ours or
another package's -- serves both, and the bundled copy is loaded only
when no other is.

Usage: share_libomp.py WHEEL_DIR

Rewrites, in place, the one wheel in WHEEL_DIR (cibuildwheel's
repair-wheel-command dest_dir, after delocate-wheel; see pyproject.toml):
its binaries, re-signed ad hoc, and its RECORD. Fails if no binary loads
the bundled libomp, so that a change in how the wheel is built cannot
silently skip this.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import base64
import csv
import hashlib
import io
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

LIBOMP = "libomp.dylib"
SHARED_NAME = "@rpath/" + LIBOMP
# A load command pointing at a bundled libomp by path, as delocate writes them
BUNDLED = re.compile(r"^\s*(@loader_path/(\S*/)?\.dylibs/" + re.escape(LIBOMP) + r")\s")


def run(*args: str) -> str:
    """Run a command, returning its standard output; fail on error."""
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def is_macho(path: pathlib.Path) -> bool:
    """Whether a file is a Mach-O binary (thin or universal)."""
    with path.open("rb") as f:
        magic = f.read(4)
    return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


def rpaths(path: pathlib.Path) -> list[str]:
    """The LC_RPATH entries of a binary."""
    out = run("otool", "-l", str(path))
    return re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", out)


def share(root: pathlib.Path) -> list[pathlib.Path]:
    """
    Rewrite the unpacked wheel under root; return the files changed.
    """
    changed: list[pathlib.Path] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and is_macho(p)):
        if path.name == LIBOMP:
            continue
        loads = run("otool", "-L", str(path)).splitlines()[1:]
        old = next((m.group(1) for line in loads if (m := BUNDLED.match(line))), None)
        if old is None:
            continue
        rpath = old[: -len("/" + LIBOMP)]
        args = ["install_name_tool", "-change", old, SHARED_NAME]
        if rpath not in rpaths(path):
            args += ["-add_rpath", rpath]
        run(*args, str(path))
        changed.append(path)
    if not changed:
        sys.exit(f"share_libomp.py: no binary in the wheel loads a bundled {LIBOMP}")

    bundled = sorted(root.rglob(".dylibs/" + LIBOMP))
    if len(bundled) != 1:
        sys.exit(f"share_libomp.py: expected one bundled {LIBOMP}, found {len(bundled)}")
    run("install_name_tool", "-id", SHARED_NAME, str(bundled[0]))
    changed.append(bundled[0])

    # install_name_tool invalidates signatures, which arm64 requires
    for path in changed:
        run("codesign", "--force", "--sign", "-", str(path))
    return changed


def record_hash(data: bytes) -> str:
    """A file's hash as a wheel RECORD writes it."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def rewrite(wheel: pathlib.Path) -> None:
    """Rewrite one wheel in place."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        with zipfile.ZipFile(wheel) as zf:
            infos = zf.infolist()
            zf.extractall(root)
        changed = share(root)

        # Update the changed files' entries in RECORD
        record_name = next(i.filename for i in infos if i.filename.endswith(".dist-info/RECORD"))
        names = {p.relative_to(root).as_posix() for p in changed}
        rows = list(csv.reader(io.StringIO((root / record_name).read_text())))
        for row in rows:
            if row and row[0] in names:
                data = (root / row[0]).read_bytes()
                row[1], row[2] = record_hash(data), str(len(data))
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(rows)
        (root / record_name).write_text(out.getvalue())

        # Repack, keeping each entry's name, order, and attributes
        new = wheel.with_suffix(".tmp")
        with zipfile.ZipFile(new, "w", zipfile.ZIP_DEFLATED) as zf:
            for info in infos:
                zf.writestr(info, (root / info.filename).read_bytes()
                            if not info.is_dir() else b"")
        new.replace(wheel)
    for name in sorted(names):
        print(f"share_libomp.py: {wheel.name}: rewrote {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("wheel_dir", type=pathlib.Path,
                        help="directory holding the one delocated wheel to rewrite")
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        sys.exit(f"share_libomp.py: expected one wheel in {args.wheel_dir}, found {len(wheels)}")
    rewrite(wheels[0])


if __name__ == "__main__":
    main()
