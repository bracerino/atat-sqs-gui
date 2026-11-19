#!/usr/bin/env bash
# =============================================================================
#  install_atat.sh  —  one-shot installer for ATAT (Alloy Theoretic Automated
#  Toolkit, incl. mcsqs) on Linux / WSL2.  Provided by the SimplySQS app.
#
#  What it does:
#    1. installs the build prerequisites (tcsh, make, g++, wget) via apt-get
#       if they are missing (asks for your sudo password),
#    2. downloads the ATAT source tarball,
#    3. compiles it and installs the binaries into <install dir>/build,
#    4. adds that build directory to your PATH in ~/.bashrc (and ~/.zshrc),
#    5. tests that `mcsqs` runs,
#    6. writes the SimplySQS example SQS files (see below).
#
#  Usage:
#    bash install_atat.sh                 # installs into ./atat3_52 next to this script
#    bash install_atat.sh /opt/my/atat    # installs into a custom directory
#  When downloaded from SimplySQS, the app's monitor.sh for an example alloy is
#  also written next to the installation (./atat_sqs_example/monitor.sh).
#
#  Options (environment variables):
#    ATAT_VERSION=3_52   ATAT version to download (default: 3_52)
#    ATAT_JOBS=N         parallel compile jobs (default: number of CPU cores)
#    ATAT_NO_PATH=1      do not modify ~/.bashrc / ~/.zshrc
# =============================================================================
set -euo pipefail

ATAT_VERSION="${ATAT_VERSION:-3_52}"
# Default location: the folder where this script was placed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${1:-$SCRIPT_DIR/atat${ATAT_VERSION}}"
JOBS="${ATAT_JOBS:-$(nproc 2>/dev/null || echo 4)}"
TARBALL="atat${ATAT_VERSION}.tar.gz"
URLS=(
    "https://axelvandewalle.github.io/www-avdw/atat/${TARBALL}"
    "http://alum.mit.edu/www/avdw/atat/${TARBALL}"
)

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✔\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m✘  %s\033[0m\n' "$*" >&2; exit 1; }
sep()  { printf '\n\033[1;36m%s\033[0m\n' "================================================================================"; }

# Resolve to an absolute path (the directory itself need not exist yet).
mkdir -p "$(dirname "$INSTALL_DIR")"
INSTALL_DIR="$(cd "$(dirname "$INSTALL_DIR")" && pwd)/$(basename "$INSTALL_DIR")"
BINDIR="$INSTALL_DIR/build"

[ -e "$INSTALL_DIR" ] && die "$INSTALL_DIR already exists. Remove it or pass another directory: bash $0 /path/to/new/atat"

# ---------------------------------------------------------------- 1. prereqs
info "Checking prerequisites (tcsh, make, g++, wget/curl)…"
missing=()
command -v tcsh >/dev/null 2>&1 || missing+=(tcsh)
command -v make >/dev/null 2>&1 || missing+=(make)
command -v g++  >/dev/null 2>&1 || missing+=(g++)
if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    missing+=(wget)
fi
if [ ${#missing[@]} -gt 0 ]; then
    if command -v apt-get >/dev/null 2>&1; then
        info "Installing missing packages: ${missing[*]} (sudo password may be requested)"
        SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
        $SUDO apt-get update
        $SUDO apt-get install -y "${missing[@]}"
    else
        die "Missing: ${missing[*]}. Please install them with your package manager and re-run."
    fi
fi
ok "Prerequisites present"

# --------------------------------------------------------------- 2. download
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
info "Downloading ATAT ${ATAT_VERSION//_/.}…"
downloaded=0
for url in "${URLS[@]}"; do
    echo "    trying $url"
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "$WORK/$TARBALL" "$url" && downloaded=1
    else
        curl -fsSL -o "$WORK/$TARBALL" "$url" && downloaded=1
    fi
    [ "$downloaded" = 1 ] && gzip -t "$WORK/$TARBALL" 2>/dev/null && break
    downloaded=0
done
[ "$downloaded" = 1 ] || die "Could not download $TARBALL. Check your internet connection or the version (ATAT_VERSION)."
ok "Downloaded $TARBALL"

tar xzf "$WORK/$TARBALL" -C "$WORK"
[ -d "$WORK/atat" ] || die "Unexpected tarball layout (no atat/ directory)."
mv "$WORK/atat" "$INSTALL_DIR"
mkdir -p "$BINDIR"
ok "Extracted to $INSTALL_DIR"

# ---------------------------------------------------------------- 3. compile
cd "$INSTALL_DIR"
# Point BINDIR in the makefile to the local build directory (so a later manual
# `make install` also works).
sed -i "1s|^BINDIR=.*|BINDIR=$BINDIR/|" makefile

# ATAT's scripts need /bin/csh. If only tcsh is available, use ATAT's own patch.
if [ ! -e /bin/csh ] && [ -x ./patchcsh ]; then
    info "/bin/csh not found — patching ATAT scripts to use tcsh"
    ./patchcsh
fi

info "Compiling ATAT with $JOBS parallel jobs (this takes a few minutes)…"
make -j "$JOBS" BINDIR="$BINDIR/"
info "Installing binaries into $BINDIR…"
make install BINDIR="$BINDIR/"
sep
ok "Compilation finished"

# ------------------------------------------------------------------- 4. PATH
export PATH="$BINDIR:$PATH"
if [ "${ATAT_NO_PATH:-0}" != 1 ]; then
    line="export PATH=\"$BINDIR:\$PATH\"  # ATAT (added by install_atat.sh)"
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ "$rc" = "$HOME/.bashrc" ] || [ -f "$rc" ]; then
            if ! grep -qsF "$BINDIR" "$rc"; then
                printf '\n%s\n' "$line" >> "$rc"
                ok "Added $BINDIR to PATH in $rc"
            else
                ok "$rc already contains $BINDIR"
            fi
        fi
    done
fi

# ------------------------------------------------------------------- 5. test
info "Testing the installation…"
command -v mcsqs >/dev/null 2>&1 || die "mcsqs not found in $BINDIR — compilation probably failed (see the output above)."
# Capture the help text first (piping straight into `grep -q` can SIGPIPE mcsqs
# and, with pipefail, report a false failure).
test_out="$(mcsqs -h 2>&1 || true)"
if printf '%s' "$test_out" | grep -qi "quasirandom"; then
    ok "mcsqs runs: $(command -v mcsqs)"
else
    die "mcsqs was installed but did not run correctly. Output of 'mcsqs -h':
$(printf '%s\n' "$test_out" | head -20)"
fi

# ------------------------------------------------- 6. example SQS calculation
# When downloaded from SimplySQS, the line below is replaced by the all-in-one
# monitor.sh that the app generates for the selected "Load example alloy"
# option; it is written to ./atat_sqs_example.
EXAMPLE_DIR=""
# @@SIMPLYSQS_EXAMPLE@@

sep
ok "ATAT ${ATAT_VERSION//_/.} is installed in $INSTALL_DIR"
echo "    Open a new terminal (or run: source ~/.bashrc) and type 'mcsqs' to start."
if [ -n "$EXAMPLE_DIR" ]; then
    echo
    echo "    Try the SimplySQS example ($EXAMPLE_LABEL):"
    echo "        cd $EXAMPLE_DIR && bash monitor.sh"
    echo "    (stop it with Ctrl+C; the best SQS so far is in bestsqs.out)"
fi
sep
