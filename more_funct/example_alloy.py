"""
One-click example / demo loader for SimplySQS.

Renders, on the landing page next to the "upload a structure" prompt, a small
crystal-structure selector (bcc / fcc / sc / hcp) plus a button that loads a
ready-made random alloy of that lattice and auto-fills *every* SQS parameter,
so a new user can immediately see a complete workflow without uploading
anything or tuning settings.

Each option is a real (or, for sc, clearly illustrative) equiatomic solid
solution whose composition divides the 3x3x3 sublattice evenly, so the target
concentrations come out exact:

    bcc -> MoNbTa   (refractory HEA)          a = 3.24 A       18 atoms each
    fcc -> CoCrFeNi (Cantor-family HEA)        a = 3.57 A       27 atoms each
    hcp -> TiZrHf   (group-4 hcp alloy)        a = 3.15, c=5.02 18 atoms each
    sc  -> VNbTa    (illustrative only *)      a = 3.20 A        9 atoms each

    * simple-cubic metallic solid solutions do not really occur (alpha-Po is the
      only sc element); the sc option is provided purely for demonstration.

For every option: sublattice-specific mode, 3x3x3 supercell, equiatomic
concentrations (from the per-element slider defaults), pair cutoff 1.5 and
triplet cutoff 1.2. After the parameters are seeded, the ATAT input files
(rndstr.in / sqscell.out), the bash commands and the all-in-one monitor.sh are
generated automatically via one-shot flags that render_atat_sqs_section
(more_funct/atat_module.py) honours on the following reruns.

Everything is done by pre-seeding the Streamlit ``session_state`` keys that the
workflow widgets read from, then letting the natural post-callback rerun rebuild
the UI with those values in place.
"""

import streamlit as st
from pymatgen.core import Structure, Lattice

# --- Example definitions ----------------------------------------------------
EXAMPLE_SUPERCELL = (3, 3, 3)
EXAMPLE_PAIR_CUTOFF = 1.5
EXAMPLE_TRIPLET_CUTOFF = 1.2

# Ordered so the selector lists them bcc, fcc, sc, hcp.
EXAMPLES = {
    "bcc": {
        "label": "BCC · MoNbTa (refractory HEA)",
        "name": "example_bcc_MoNbTa.cif",
        "elements": ["Mo", "Nb", "Ta"],
        "a": 3.24,
    },
    "fcc": {
        "label": "FCC · CoCrFeNi (Cantor-type HEA)",
        "name": "example_fcc_CoCrFeNi.cif",
        "elements": ["Co", "Cr", "Fe", "Ni"],
        "a": 3.57,
    },
    "sc": {
        "label": "SC · VNbTa (illustrative)",
        "name": "example_sc_VNbTa.cif",
        "elements": ["V", "Nb", "Ta"],
        "a": 3.20,
    },
    "hcp": {
        "label": "HCP · TiZrHf (group-4 hcp)",
        "name": "example_hcp_TiZrHf.cif",
        "elements": ["Ti", "Zr", "Hf"],
        "a": 3.15,
        "c": 5.02,
    },
}


def create_example_structure(kind):
    """Build the small conventional base cell for the requested lattice.

    Sublattice-specific mode replaces every site with the selected elements, so
    the base occupancy only has to define the crystallographic *positions*; the
    alloy's first element is used as a placeholder occupant.
    """
    info = EXAMPLES[kind]
    base = info["elements"][0]
    a = info["a"]

    if kind == "sc":                                  # simple cubic, 1 atom
        return Structure(Lattice.cubic(a), [base], [[0.0, 0.0, 0.0]])
    if kind == "bcc":                                 # body-centred cubic, 2 atoms
        return Structure(Lattice.cubic(a), [base] * 2,
                         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    if kind == "fcc":                                 # face-centred cubic, 4 atoms
        return Structure(Lattice.cubic(a), [base] * 4,
                         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    if kind == "hcp":                                 # hexagonal close packed, 2 atoms
        return Structure(Lattice.hexagonal(a, info["c"]), [base] * 2,
                         [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])
    raise ValueError(f"Unknown structure kind: {kind}")


def load_example_alloy():
    """Inject the selected example structure and pre-set all SQS parameters.

    Used as an ``on_click`` callback (Streamlit reruns automatically afterwards,
    so we must not call ``st.rerun`` here). On that rerun the workflow widgets
    are created for the first time and pick up the values seeded below. The
    crystal-structure choice is read from the selector's session-state key.
    """
    kind = st.session_state.get("example_alloy_kind", "bcc")
    info = EXAMPLES[kind]

    if not st.session_state.get("full_structures"):
        st.session_state.full_structures = {}
    st.session_state.full_structures[info["name"]] = create_example_structure(kind)

    nx, ny, nz = EXAMPLE_SUPERCELL
    preset = {
        # structure selection inside render_atat_sqs_section
        "atat_structure_selector": info["name"],
        "atat_reduce_primitive": False,
        # Step 1 - composition mode (sublattice-specific)
        "atat_composition_mode_radio": "🎯 Sublattice-Specific (Recommended)",
        # Step 2 - supercell
        "atat_nx": nx,
        "atat_ny": ny,
        "atat_nz": nz,
        # Step 3 - elements on the single Wyckoff sublattice "A".
        # (Equiatomic concentrations follow from the per-element slider defaults.)
        "default_sublattice_A_elements_v2": list(info["elements"]),
        # Step 4 - cluster cutoffs: pairs + triplets on, quadruplets off
        "atat_pair_cutoff": EXAMPLE_PAIR_CUTOFF,
        "atat_include_triplets": True,
        "atat_triplet_cutoff_val": EXAMPLE_TRIPLET_CUTOFF,
        "atat_include_quadruplets": False,
    }
    for key, value in preset.items():
        st.session_state[key] = value

    # One-shot flags: auto-generate the ATAT input files + bash commands and the
    # all-in-one monitor.sh on the upcoming reruns, as if the buttons were pressed.
    st.session_state["example_auto_generate"] = True
    st.session_state["example_auto_generate_monitor"] = True
    st.session_state["example_alloy_loaded"] = True


def render_example_selector():
    """Selector for the example crystal-structure type (bcc / fcc / sc / hcp)."""
    st.selectbox(
        "Example crystal structure:",
        options=list(EXAMPLES.keys()),
        format_func=lambda k: EXAMPLES[k]["label"],
        key="example_alloy_kind",
    )


def render_example_alloy_button():
    """Button that loads the currently-selected example alloy in one click."""
    kind = st.session_state.get("example_alloy_kind", "bcc")
    info = EXAMPLES[kind]
    formula = "".join(info["elements"])
    # Small spacer so the button lines up with the selectbox input (which has a label).
    st.markdown("<div style='height:1.7em'></div>", unsafe_allow_html=True)
    st.button(
        f"🎲 Load example alloy — {kind.upper()} · {formula} (3×3×3)",
        type="primary",
        use_container_width=True,
        key="load_example_alloy_btn",
        on_click=load_example_alloy,
    )
