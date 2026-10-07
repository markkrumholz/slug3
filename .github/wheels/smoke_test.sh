#!/usr/bin/env bash
#
# Smoke test an installed (pip-installed or wheel-installed) copy of slug:
# runs a small cluster simulation with yields both through the slug console
# script and through slugpy.run_sim, then reads both outputs back with
# slugpy (and therefore h5py, whose own HDF5 is loaded into the same
# process as slug's).
#
# Runs from a scratch directory, outside the source tree, with SLUG_DIR
# and PYTHONPATH unset, so the IMF, yield tables, isotope table, and
# SimControls' Python defaults can only be found via the installed
# package's own bundled data (SLUG_DATA_PATH -- see slugpy/_paths.py). The
# only thing it borrows from the source tree is the small test track
# fixture, by absolute path, since real track libraries are far too large
# for CI.
#
# Usage: smoke_test.sh REPO_DIR
#
# Used by the pip-install job in .github/workflows/ci.yml and by
# cibuildwheel's test-command (see pyproject.toml).
#
# :copyright: Copyright (c) 2026 Mark Krumholz

set -euo pipefail

REPO="$(cd "$1" && pwd)"
unset SLUG_DIR PYTHONPATH
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
cd "${WORK}"

cat > deck.toml <<EOF
sim_type = "cluster"
n_trial = 3

[stars]
IMF = "chabrier.toml"
track_registry = "${REPO}/tests/tracks/assets/tracks.toml"
tracks = "MIST_test"
FeH = 0.0
alphaFe = -0.2

[spectra]
model = "blackbody"

[clusters]
CMF = 1e3

[output]
model_name = "smoke_cli"
output_times = [1.0e6, 1.0e7]

[yields.channel1]
channel = "ccsn"
model = "sukhbold16"

[nebular]
compute_neb = false
EOF
sed 's/smoke_cli/smoke_py/' deck.toml > deck_py.toml

slug deck.toml
python - <<'EOF'
import slugpy

for name in ("smoke_cli", "smoke_py"):
    if name == "smoke_py":
        slugpy.run_sim(slugpy.SimControls("deck_py.toml"), progress=False)
    r = slugpy.read(name)
    assert len(r.clusters["uid"]) == 3, name
    assert len(r.cluster_yields.isotopes) > 0, name
    print(name, "n_sn:", r.cluster_feedback["n_sn"][:])
print("smoke test passed, using slugpy from", slugpy.__file__)
EOF
