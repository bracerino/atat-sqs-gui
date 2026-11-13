"""Structures pasted as text (POSCAR, CIF, extended XYZ, LAMMPS data) in the sidebar.

Each pasted structure is kept in the session state and exposed as a file-like
object with a ``name`` so the rest of the app treats it like an uploaded file.
"""
import io
import os
import re
import tempfile

import streamlit as st

FORMATS = {
    "Auto-detect": None,
    "POSCAR (VASP)": ".vasp",
    "CIF": ".cif",
    "Extended XYZ": ".xyz",
    "LAMMPS data": ".lmp",
}


class PastedStructureFile(io.BytesIO):
    """In-memory stand-in for a Streamlit UploadedFile."""

    def __init__(self, name, content):
        super().__init__(content.encode("utf-8"))
        self.name = name
        self.file_id = f"pasted::{name}"
        self.size = len(content)


def detect_extension(text):
    lowered = text.lower()
    if re.search(r"^\s*data_", lowered, re.MULTILINE) or "_cell_length_a" in lowered:
        return ".cif"
    if re.search(r"\bxlo\s+xhi\b", lowered) and "atoms" in lowered:
        return ".lmp"
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first_line.isdigit():
        return ".xyz"
    return ".vasp"


def _parse(content, extension):
    """Parse the text to check it is a valid structure; raises on failure."""
    from ase.io import read
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.lammps.data import LammpsData

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pasted" + extension)
        with open(path, "w") as f:
            f.write(content)
        if extension == ".cif":
            return Structure.from_file(path)
        if extension == ".lmp":
            return LammpsData.from_file(path, atom_style="atomic").structure
        return AseAtomsAdaptor.get_structure(read(path))


def _add_pasted_structure():
    content = st.session_state.get("paste_structure_text", "")
    if not content.strip():
        st.session_state["paste_structure_message"] = ("warning", "Paste the structure content first.")
        return
    extension = FORMATS[st.session_state.get("paste_structure_format", "Auto-detect")] or detect_extension(content)
    try:
        structure = _parse(content, extension)
    except Exception as e:
        st.session_state["paste_structure_message"] = (
            "error", f"Could not read the pasted content as {extension.lstrip('.').upper()}: {e}")
        return

    pasted = st.session_state.setdefault("pasted_structures", {})
    base = re.sub(r"[^\w\-.]", "_", st.session_state.get("paste_structure_name", "").strip())
    if not base:
        base = structure.composition.reduced_formula
    base = base if base.startswith("pasted_") else f"pasted_{base}"
    if base.lower().endswith(extension):
        base = base[: -len(extension)]
    name, i = f"{base}{extension}", 2
    while name in pasted:
        name, i = f"{base}_{i}{extension}", i + 1

    pasted[name] = content
    st.session_state["paste_structure_text"] = ""
    st.session_state["paste_structure_name"] = ""
    st.session_state["paste_structure_message"] = (
        "success", f"Added {name} ({structure.composition.reduced_formula}, {len(structure)} atoms).")


def render_paste_structure_input():
    st.text_area(
        "Paste the content of a structure file (POSCAR, CIF, extended XYZ, LAMMPS data):",
        key="paste_structure_text",
        height=200,
        placeholder="Fe\n1.0\n2.87 0 0\n0 2.87 0\n0 0 2.87\nFe\n2\nDirect\n0 0 0\n0.5 0.5 0.5",
    )
    col_format, col_name = st.columns(2)
    col_format.selectbox("Format", list(FORMATS), key="paste_structure_format")
    col_name.text_input("Name (optional)", key="paste_structure_name")
    st.button("➕ Add structure", on_click=_add_pasted_structure,
              key="paste_structure_add", use_container_width=True)

    message = st.session_state.pop("paste_structure_message", None)
    if message:
        getattr(st, message[0])(message[1])

    pasted = st.session_state.get("pasted_structures", {})
    for name in list(pasted):
        col_label, col_remove = st.columns([4, 1])
        col_label.text(f"• {name}")
        if col_remove.button("🗑️", key=f"remove_pasted_{name}", help=f"Remove {name}"):
            del pasted[name]
            st.rerun()


def get_pasted_structure_files():
    return [PastedStructureFile(name, content)
            for name, content in st.session_state.get("pasted_structures", {}).items()]
