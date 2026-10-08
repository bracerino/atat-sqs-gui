"""
Save / load the whole SQS setup as one JSON file.

The file holds the selected initial structure (as a pymatgen dict, so nothing is
lost to CIF rounding) and the values of every widget of the generation workflow:
primitive-cell reduction, composition mode, supercell, global or per-sublattice
elements and concentrations, cluster cutoffs and the monitor.sh options.

Loading works like the "Load example alloy" button (more_funct/example_alloy.py):
the structure is put into ``full_structures`` and the widget keys are pre-seeded
in ``session_state`` when the file is dropped into the structure uploader, which
runs before any of those widgets is created. The ATAT input files are then
regenerated through the same one-shot ``example_auto_generate`` flag.

Only plain widget keys are stored. Buttons must never be included: Streamlit
refuses to set a button's value through ``session_state``.
"""

import json
import re
from datetime import datetime, timezone

import streamlit as st
from monty.json import MontyEncoder
from pymatgen.core import Structure

SETTINGS_FORMAT = "SimplySQS-settings"
SETTINGS_FORMAT_VERSION = 1

SETTINGS_KEYS = {
    # structure selection
    "atat_structure_selector",
    "atat_reduce_primitive",
    # Step 1 - composition mode
    "atat_composition_mode_radio",
    # Step 2 - supercell
    "atat_nx", "atat_ny", "atat_nz",
    # Step 3 - global composition
    "atat_composition_global",
    "atat_global_conc_basis",
    "atat_comp_global_conc_input_mode",
    # Step 4 - cluster cutoffs
    "atat_pair_cutoff",
    "atat_include_triplets", "atat_triplet_cutoff_val",
    "atat_include_quadruplets", "atat_quadruplet_cutoff_val",
    # monitor.sh options (quick panel + advanced section)
    "quick_monitor_panel",
    "quick_monitor_execution_mode", "quick_monitor_enable_parallel", "quick_monitor_parallel_count",
    "quick_monitor_enable_time_limit", "quick_monitor_time_limit",
    "monitor_execution_mode", "monitor_enable_parallel", "monitor_parallel_count",
    "monitor_enable_time_limit", "monitor_time_limit",
}

# Keys built at runtime (element names, sublattice letters, concentration denominators).
SETTINGS_KEY_PATTERNS = [
    re.compile(r"^atat_comp_global_.+_\d+$"),                       # global sliders / number inputs
    re.compile(r"^default_(separate_by_coords|conc_input_mode)$"),  # Step 3 sublattice toggles
    re.compile(r"^default_sublattice_.+_elements_v2$"),             # sublattice elements
    re.compile(r"^default_sublattice_.+_(frac|num)_v2$"),           # sublattice concentrations
]


def _is_settings_key(key):
    return key in SETTINGS_KEYS or any(p.match(key) for p in SETTINGS_KEY_PATTERNS)


def _to_plain(value):
    """Widget value as plain JSON (numpy scalars -> Python), or None if unsupported."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, (bool, int, float, str)) for v in value):
        return list(value)
    return None


def build_settings_json():
    """JSON text with the selected structure and current workflow settings, or None."""
    structures = st.session_state.get("full_structures") or {}
    name = st.session_state.get("atat_structure_selector")
    if name not in structures:
        name = next(iter(structures), None)
    if name is None:
        return None, None

    settings = {}
    for key in sorted(st.session_state.keys()):
        if _is_settings_key(key):
            value = _to_plain(st.session_state[key])
            if value is not None:
                settings[key] = value
    settings["atat_structure_selector"] = name

    data = {
        "format": SETTINGS_FORMAT,
        "format_version": SETTINGS_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "structure": {"name": name, "pymatgen": structures[name].as_dict()},
        "settings": settings,
    }
    return name, json.dumps(data, indent=2, cls=MontyEncoder)


def apply_settings_data(data):
    """Put the saved structure into full_structures and seed the widget values.

    Must run before the workflow widgets are created (see split_settings_files), otherwise
    Streamlit refuses to change their values.
    """
    if data.get("format") != SETTINGS_FORMAT:
        raise ValueError("this is not a SimplySQS settings file")
    if data.get("format_version", 0) > SETTINGS_FORMAT_VERSION:
        raise ValueError("the file was saved by a newer version of SimplySQS")

    name = data["structure"]["name"]
    structure = Structure.from_dict(data["structure"]["pymatgen"])
    if not st.session_state.get("full_structures"):
        st.session_state.full_structures = {}
    st.session_state.full_structures[name] = structure

    for key, value in data.get("settings", {}).items():
        if _is_settings_key(key):
            st.session_state[key] = value
    st.session_state["atat_structure_selector"] = name

    # Regenerate rndstr.in / sqscell.out / commands as if the button was pressed.
    st.session_state["example_auto_generate"] = True
    st.session_state["atat_results"] = None
    return name


def _read_settings_file(uploaded):
    """Parsed settings dict if ``uploaded`` is a SimplySQS settings file, otherwise None."""
    if not uploaded.name.lower().endswith(".json"):
        return None
    try:
        data = json.loads(uploaded.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == SETTINGS_FORMAT else None


def split_settings_files(uploaded_files):
    """Apply settings files dropped into the structure uploader; return the other files.

    Must be called before the workflow widgets are created. Each upload is applied
    only once (tracked by its file_id), so later widget changes are not overwritten
    while the file stays in the uploader. Removing and re-adding it loads it again.
    """
    applied = st.session_state.setdefault("applied_settings_file_ids", set())
    messages = st.session_state.setdefault("settings_load_messages", {})
    structure_files, present = [], set()
    for uploaded in uploaded_files or []:
        data = _read_settings_file(uploaded)
        if data is None:
            structure_files.append(uploaded)
            continue
        present.add(uploaded.file_id)
        if uploaded.file_id in applied:
            continue
        applied.add(uploaded.file_id)
        try:
            name = apply_settings_data(data)
            messages[uploaded.file_id] = (
                "success", f"Loaded settings and structure **{name}** from {uploaded.name}.")
        except Exception as e:
            messages[uploaded.file_id] = ("error", f"Could not load the settings file {uploaded.name}: {e}")
    # Keep each message while its file stays in the uploader (loading triggers an
    # immediate rerun for the auto-generation, so a one-shot message would vanish).
    for file_id in list(messages):
        if file_id not in present:
            del messages[file_id]
    return structure_files


def render_settings_save_load():
    """Sidebar block directly below the structure uploader: a short note on loading
    settings files, then the button that saves the current setup.

    A saved file is loaded back through the structure uploader (split_settings_files).
    """
    st.sidebar.caption(
        "💡 You can also upload a saved **SimplySQS settings file (.json)** here."
    )
    name, settings_json = build_settings_json()
    if settings_json is None:
        st.sidebar.button(
            "💾 Save current settings",
            disabled=True,
            type="primary",
            width="stretch",
            key="settings_download_btn_disabled",
        )
    else:
        stem = name.rsplit(".", 1)[0]
        st.sidebar.download_button(
            "💾 Save current settings",
            data=settings_json,
            file_name=f"SimplySQS_settings_{stem}.json",
            mime="application/json",
            type="primary",
            width="stretch",
            key="settings_download_btn",
        )

    for kind, text in st.session_state.get("settings_load_messages", {}).values():
        (st.sidebar.success if kind == "success" else st.sidebar.error)(text)
