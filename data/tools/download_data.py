"""
Download slug's large data files (stellar tracks, spectral libraries,
filters, etc.) from the public mirror on the ANU Data Commons, and
install them into the matching subdirectories of data/. See --help
for the full detail (merge rules, etc.); see also AGENTS.md's own
Data files section.

:copyright: Copyright (c) 2026 Mark Krumholz
"""

import argparse
import base64
import binascii
import hashlib
import pathlib
import re

import tomlkit
import urllib3

DEFAULT_BASE_URL = "https://datacommons.anu.edu.au/DataCommons/rest/records/anudc:6518/data/"
# data/tools/download_data.py -> data/ is one directory up
DEFAULT_DEST = pathlib.Path(__file__).resolve().parent.parent

# The mirror's directory listing embeds each entry (file or
# subdirectory) as a checkbox value, e.g.
#     <input type="checkbox" name="i" value="filters.h5" />
# for a file, or value="spectra/" (trailing slash) for a subdirectory.
# This is the same "scrape an href/value out of raw HTML with a
# targeted regex" approach fetch_mist.py already uses on a different
# server's directory listing.
_ENTRY_RE = re.compile(r'name="i"\s+value="([^"]+)"')


_EPILOG = """
The mirror's own directory structure is a direct copy of data/'s own:
top-level entries are data subdirectories (e.g. filters, spectra,
tracks), each containing that subdirectory's *.h5 library files plus
a *.toml registry indexing them. By default, every subdirectory the
mirror actually has is downloaded; --subdir restricts this to just
one, e.g. for a smaller test download or to refresh a single library.

Existing files are left alone by default -- pass --overwrite to
redownload them anyway. This applies differently to *.toml registry
files than to everything else: a *.toml file already present locally
is never simply replaced, since it may hold entries a user added by
hand (e.g. via the various fetch_*.py/add_extinction_curve.py scripts
in this directory) that the mirror knows nothing about. Instead, its
content is merged with the downloaded version, key by key -- a table
present in both is merged recursively (so entries only one side has
survive), a list present in both is unioned (order-preserving, mirror
entries first), and anything else is taken from the mirror. Only
--overwrite makes a *.toml file a plain wholesale replacement, the
same as for every other file type.

Run from anywhere, e.g. from the repository root:
    python3 data/tools/download_data.py
    python3 data/tools/download_data.py --subdir filters
    python3 data/tools/download_data.py --overwrite --subdir tracks
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
        help=f"Base URL of the data mirror (default: {DEFAULT_BASE_URL})")
    p.add_argument("--dest", default=str(DEFAULT_DEST),
        help="Local data/ directory to install into "
             f"(default: {DEFAULT_DEST}, i.e. this repository's own data/)")
    p.add_argument("--subdir",
        help="Only download this one subdirectory (e.g. filters, spectra, "
             "tracks) instead of everything the mirror has")
    p.add_argument("--overwrite", action="store_true",
        help="Redownload files that already exist locally, instead of "
             "skipping them; also makes *.toml registries a wholesale "
             "replacement instead of a merge (see epilog)")
    p.add_argument("--verbose", action="store_true",
        help="Print progress for every file, not just a final summary")
    return p.parse_args()


def list_remote_entries(url: str, http: urllib3.PoolManager) -> list[str]:
    """Return the file/subdirectory names (mirror's own checkbox values) listed at url."""
    resp = http.request("GET", url)
    if resp.status != 200:
        raise RuntimeError(f"Failed to list {url}: HTTP {resp.status}")
    return _ENTRY_RE.findall(resp.data.decode("utf-8", errors="replace"))


def _digest_matches(digest: bytes, content_md5: str) -> bool:
    """True if content_md5 (a Content-MD5 response header value) matches digest,
    a raw MD5 digest (e.g. from hashlib.md5(...).digest()).

    RFC 1864 specifies Content-MD5 as base64, but this mirror actually
    sends a plain hex digest; both forms are accepted here rather than
    assuming one, so this keeps working even if that ever changes. A
    value that is neither is treated as a verification failure (not
    skipped), so a corrupted download can never slip through simply
    because its own integrity header came back unparseable.
    """
    content_md5 = content_md5.strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", content_md5):
        return content_md5.lower() == digest.hex()
    try:
        decoded = base64.b64decode(content_md5, validate=True)
    except (binascii.Error, ValueError):
        return False
    return decoded == digest


def download_bytes(url: str, http: urllib3.PoolManager) -> bytes:
    """GET url and return its content, verifying Content-MD5 if the server sends one.

    Loads the whole response into memory -- fine for the small *.toml
    registries this is used for, but see stream_download() for the
    (potentially multi-GB) *.h5 library files.
    """
    resp = http.request("GET", url)
    if resp.status != 200:
        raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.status}")
    content_md5 = resp.headers.get("Content-MD5")
    if content_md5 and not _digest_matches(hashlib.md5(resp.data).digest(), content_md5):
        raise RuntimeError(f"{url}: downloaded content does not match its own Content-MD5 header")
    return resp.data


def stream_download(url: str, dest: pathlib.Path, http: urllib3.PoolManager) -> int:
    """Stream url's content straight to dest (some of these files are multiple
    GB, so this never holds a whole file in memory at once), verifying Content-MD5
    if the server sends one. Written to a "dest.part" sibling first and only
    renamed into place on success, so a download interrupted partway through
    (network failure, Ctrl-C) never leaves a corrupt file sitting at dest itself
    for a future run to mistake for a already-downloaded one. Returns the number
    of bytes written.
    """
    resp = http.request("GET", url, preload_content=False)
    if resp.status != 200:
        resp.release_conn()
        raise RuntimeError(f"Failed to fetch {url}: HTTP {resp.status}")

    tmp = dest.with_name(dest.name + ".part")
    hasher = hashlib.md5()
    nbytes = 0
    try:
        try:
            with open(tmp, "wb") as f:
                for chunk in resp.stream(1 << 20):
                    hasher.update(chunk)
                    f.write(chunk)
                    nbytes += len(chunk)
        finally:
            resp.release_conn()
    except BaseException:
        # Covers a network failure partway through as well as a
        # Ctrl-C (KeyboardInterrupt, not an Exception subclass) --
        # either way, tmp holds an incomplete file that should not
        # linger around as if it were a real, resumable download.
        tmp.unlink(missing_ok=True)
        raise

    content_md5 = resp.headers.get("Content-MD5")
    if content_md5 and not _digest_matches(hasher.digest(), content_md5):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: downloaded content does not match its own Content-MD5 header")

    tmp.replace(dest)
    return nbytes


def deep_merge_toml(local: dict, remote: dict) -> None:
    """Merge remote into local in place: recurse into shared tables, union shared
    lists (remote's entries first, then any local-only ones), and otherwise take
    remote's value -- so a key only local has is never touched."""
    for key, remote_val in remote.items():
        if key in local:
            local_val = local[key]
            if isinstance(remote_val, dict) and isinstance(local_val, dict):
                deep_merge_toml(local_val, remote_val)
                continue
            if isinstance(remote_val, list) and isinstance(local_val, list):
                merged = list(remote_val)
                merged.extend(item for item in local_val if item not in merged)
                local[key] = merged
                continue
        local[key] = remote_val


def install_toml(remote_bytes: bytes, dest: pathlib.Path, overwrite: bool, verbose: bool) -> None:
    """Write remote_bytes (a downloaded *.toml registry) to dest, merging with
    whatever dest already holds unless overwrite is set (see this module's own
    docstring for the exact merge semantics)."""
    if overwrite or not dest.exists():
        dest.write_bytes(remote_bytes)
        if verbose:
            print(f"wrote {dest}")
        return

    local_doc = tomlkit.parse(dest.read_text())
    remote_doc = tomlkit.parse(remote_bytes.decode("utf-8"))
    deep_merge_toml(local_doc, remote_doc)
    dest.write_text(tomlkit.dumps(local_doc))
    if verbose:
        print(f"merged {dest}")


def _is_safe_path_component(name: str) -> bool:
    """True if name is safe to use as a single local path component.

    Rejects anything that isn't a plain basename -- an absolute path
    (which, joined onto an existing pathlib.Path with /, silently
    discards that existing path rather than erroring, so this is not
    just cosmetic), a path separator of either flavor, or a "."/".."
    traversal component -- since name ultimately comes from parsing
    the mirror's own HTML (or, for the top-level case, --subdir),
    rather than from a fixed, trusted list.
    """
    if not name or name in (".", ".."):
        return False
    return "/" not in name and "\\" not in name


def sync_directory(remote_url: str, dest_dir: pathlib.Path, http: urllib3.PoolManager,
                    overwrite: bool, verbose: bool) -> None:
    """Recursively mirror remote_url (a mirror directory listing) into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in list_remote_entries(remote_url, http):
        is_dir = name.endswith("/")
        component = name[:-1] if is_dir else name
        if not _is_safe_path_component(component):
            raise RuntimeError(
                f"Refusing unsafe entry name {name!r} listed at {remote_url}")

        if is_dir:
            sync_directory(remote_url + name, dest_dir / component, http, overwrite, verbose)
            continue

        dest = dest_dir / component
        if dest.suffix == ".toml":
            # Always fetched, even if present -- install_toml() itself
            # decides whether that means a merge or (with --overwrite)
            # a wholesale replacement.
            data = download_bytes(remote_url + name, http)
            install_toml(data, dest, overwrite, verbose)
            continue

        if dest.exists() and not overwrite:
            if verbose:
                print(f"skipping {dest} (already exists)")
            continue

        if verbose:
            print(f"downloading {remote_url}{name} -> {dest}")
        nbytes = stream_download(remote_url + name, dest, http)
        if verbose:
            print(f"wrote {dest} ({nbytes} bytes)")


def main() -> None:
    args = parse_args()
    base_url = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    dest_root = pathlib.Path(args.dest)

    http = urllib3.PoolManager()

    if args.subdir:
        if not _is_safe_path_component(args.subdir):
            raise SystemExit(f"--subdir must be a single directory name, not {args.subdir!r}")
        subdirs = [args.subdir]
    else:
        subdirs = [name.rstrip("/") for name in list_remote_entries(base_url, http)
                   if name.endswith("/")]
        if args.verbose:
            print(f"found subdirectories: {', '.join(subdirs)}")

    for subdir in subdirs:
        print(f"Syncing {subdir}...")
        sync_directory(base_url + subdir + "/", dest_root / subdir, http, args.overwrite, args.verbose)

    print("Done.")


if __name__ == "__main__":
    main()
