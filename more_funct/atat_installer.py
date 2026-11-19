"""Download button for the one-shot ATAT installer script (install_atat.sh).

The downloaded script also contains a ready-to-run example: the all-in-one
monitor.sh (it writes rndstr.in / sqscell.out itself) that SimplySQS generates
for the example alloy
currently selected next to the "Load example alloy" button (default settings
of the monitor.sh panel: -rc supercell mode, 3 parallel runs, no time limit).
"""
import os

import streamlit as st

ATAT_LATEST_VERSION = "3.52"
_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "install_atat.sh")
_PLACEHOLDER = "# @@SIMPLYSQS_EXAMPLE@@"
_MONITOR_PARALLEL_RUNS = 3   # default of the "Number of parallel runs" input


def generate_example_files(kind):
    """The all-in-one monitor.sh exactly as the app builds it for an example alloy."""
    from more_funct.example_alloy import (EXAMPLES, EXAMPLE_SUPERCELL, EXAMPLE_PAIR_CUTOFF,
                                          EXAMPLE_TRIPLET_CUTOFF, create_example_structure)
    from more_funct.atat_module import (generate_atat_rndstr_content_corrected,
                                        generate_atat_sqscell_content, generate_atat_monitor_script)

    info = EXAMPLES[kind]
    structure = create_example_structure(kind)
    elements = list(info["elements"])
    nx, ny, nz = EXAMPLE_SUPERCELL
    total_atoms = len(structure) * nx * ny * nz

    # Sublattice-specific mode, one sublattice "A" holding all sites, equiatomic.
    concentrations = {"A": {el: 1.0 / len(elements) for el in elements}}
    rndstr = generate_atat_rndstr_content_corrected(
        structure, concentrations, True, [elements] * len(structure), None)
    sqscell = generate_atat_sqscell_content(nx, ny, nz)
    lat = structure.lattice
    monitor = generate_atat_monitor_script(
        results={  # same keys the app stores in st.session_state.atat_results
            "structure_name": info["name"],
            "supercell_size": f"{nx}×{ny}×{nz}",
            "total_atoms": total_atoms,
            "pair_cutoff": EXAMPLE_PAIR_CUTOFF,
            "triplet_cutoff": EXAMPLE_TRIPLET_CUTOFF,
            "quadruplet_cutoff": None,
            "rndstr_content": rndstr,
            "sqscell_content": sqscell,
            "max_param": max(lat.a, lat.b, lat.c),
        },
        use_atom_count=False,
        parallel_runs=_MONITOR_PARALLEL_RUNS,
        pair_cutoff=EXAMPLE_PAIR_CUTOFF,
        triplet_cutoff=EXAMPLE_TRIPLET_CUTOFF,
        quadruplet_cutoff=None,
        max_param=max(lat.a, lat.b, lat.c),
        time_limit_minutes=None,
    )
    label = f"{kind.upper()} {''.join(elements)}, {nx}x{ny}x{nz} supercell, {total_atoms} atoms"
    # monitor.sh writes rndstr.in and sqscell.out itself, so it is the only file needed.
    return {"monitor.sh": monitor}, label


def _example_bash_block(files, label):
    lines = [
        f'EXAMPLE_LABEL="{label}"',
        'EXAMPLE_DIR="${ATAT_EXAMPLE_DIR:-$(dirname "$INSTALL_DIR")/atat_sqs_example}"',
        'if [ -e "$EXAMPLE_DIR" ]; then',
        '    info "$EXAMPLE_DIR already exists — not creating the SimplySQS example files."',
        '    EXAMPLE_DIR=""',
        'else',
        '    mkdir -p "$EXAMPLE_DIR"',
    ]
    for i, (name, content) in enumerate(files.items()):
        delim = f"SIMPLYSQS_EOF_{i}"
        lines.append(f"cat > \"$EXAMPLE_DIR/{name}\" <<'{delim}'")
        lines.append(content.rstrip("\n"))
        lines.append(delim)
    lines += [
        '    chmod +x "$EXAMPLE_DIR/monitor.sh"',
        '    ok "SimplySQS example ($EXAMPLE_LABEL) written to $EXAMPLE_DIR"',
        'fi',
    ]
    return "\n".join(lines)


def build_install_script(kind):
    with open(_SCRIPT_PATH, encoding="utf-8") as f:
        script = f.read()
    files, label = generate_example_files(kind)
    return script.replace(_PLACEHOLDER, _example_bash_block(files, label), 1)


def render_atat_install_script_button(key):
    """Download button for install_atat.sh (download, compile, add to PATH, test mcsqs, example)."""
    kind = st.session_state.get("example_alloy_kind", "bcc")
    try:
        script = build_install_script(kind)
    except Exception:
        # Never block the download: fall back to the installer without the example.
        with open(_SCRIPT_PATH, encoding="utf-8") as f:
            script = f.read()
    st.download_button(
        f"🛠️ Download ATAT {ATAT_LATEST_VERSION} install script (Linux/WSL2)",
        data=script.encode("utf-8"),
        file_name="install_atat.sh",
        mime="text/x-shellscript",
        type="primary",
        width="stretch",
        key=key,
        help=(
            f"Bash script that installs prerequisites (tcsh, make, g++), downloads ATAT "
            f"{ATAT_LATEST_VERSION}, compiles it into ./atat3_52 (in the folder where you place the "
            "script), adds it to PATH in ~/.bashrc and tests that mcsqs runs. It also writes the "
            "SimplySQS monitor.sh for the selected example alloy to "
            "./atat_sqs_example. Run it with: bash install_atat.sh"
        ),
    )
