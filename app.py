import streamlit as st

st.set_page_config(
    page_title="SimplySQS: Create Input Files for Generation of SQS using ATAT mcsqs and Analyse Its Outputs",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

from pymatgen.core import Structure
import io
import os
import re
from pymatgen.io.cif import CifWriter
import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



#import pkg_resources
#installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
#st.subheader("Installed Python Modules")
#for package, version in installed_packages:
#    st.write(f"{package}=={version}")

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []

if 'previous_uploaded_files' not in st.session_state:
    st.session_state['previous_uploaded_files'] = []

st.markdown(
    """
    <style>
        /* Sidebar background with soft transparent red-green gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(231, 76, 60, 0.15),   /* very soft red */
                rgba(46, 204, 113, 0.15)   /* very soft green */
            );
            backdrop-filter: blur(6px);  /* frosted glass effect */
        }

        /* Custom caption style */
        .sidebar-caption {
            font-size: 1.15rem;
            font-weight: 600;
            color: inherit;
            margin: 1rem 0 0.5rem 0;
            position: relative;
            display: inline-block;
        }

        .sidebar-caption::after {
            content: "";
            display: block;
            width: 100%;
            height: 3px;
            margin-top: 4px;
            border-radius: 2px;
            background: linear-gradient(to right, #e74c3c, #2ecc71);  /* vivid red → green underline */
        }

        /* The default open/close control is a thin grey chevron that is easy
           to miss once the sidebar is hidden, so both buttons carry the same
           red-to-green gradient as the sidebar itself (the saturated version
           used by the caption underline). stExpandSidebarButton is the one
           shown over the page when the sidebar is collapsed,
           stSidebarCollapseButton the one inside the sidebar header. */
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarCollapseButton"] button {
            background: linear-gradient(135deg, #e74c3c, #2ecc71) !important;
            color: #ffffff !important;
            border: 1px solid #ffffff !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
            opacity: 1 !important;
            transition: filter 0.15s ease, transform 0.15s ease;
        }
        [data-testid="stExpandSidebarButton"]:hover,
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: linear-gradient(135deg, #c0392b, #27ae60) !important;
            transform: scale(1.06);
        }
        /* Material icon fonts and inline SVGs both follow the button colour. */
        [data-testid="stExpandSidebarButton"] *,
        [data-testid="stSidebarCollapseButton"] button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stExpandSidebarButton"] {
            min-height: 34px !important;
            min-width: 34px !important;
            padding: 0 6px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-caption">SQS ATAT</div>', unsafe_allow_html=True)

# File uploader in the sidebar


st.sidebar.subheader("📁 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, LMP, extended XYZ):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)


current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

# Detect removed files
removed_files = set(previous_file_names) - set(current_file_names)


for removed_file in removed_files:
    if removed_file in st.session_state.full_structures:
        del st.session_state.full_structures[removed_file]
        st.success(f"Removed structure: {removed_file}")


st.session_state['uploaded_files'] = [
    file for file in st.session_state['uploaded_files']
    if file.name in current_file_names
]

import os
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
import streamlit as st


def load_structure(file):
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer

        with open(file.name, "wb") as f:
            f.write(file_content)
        filename = file.name.lower()

        if filename.endswith(".cif"):
            mg_structure = Structure.from_file(file.name)
        elif filename.endswith(".data"):
            lmp_filename = file.name.replace(".data", ".lmp")
            os.rename(file.name, lmp_filename)
            lammps_data = LammpsData.from_file(lmp_filename, atom_style="atomic")
            mg_structure = lammps_data.structure
        elif filename.endswith(".lmp"):
            lammps_data = LammpsData.from_file(file.name, atom_style="atomic")
            mg_structure = lammps_data.structure
        else:
            atoms = read(file.name)
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

        if os.path.exists(file.name):
            os.remove(file.name)

        return mg_structure

    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        st.error(
            f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
            f"and please remove this wrongly placed file. 😊")
        raise e


def handle_uploaded_files(uploaded_files_user_sidebar):

    if uploaded_files_user_sidebar:
        for file in uploaded_files_user_sidebar:
            if file.name not in st.session_state.full_structures:
                try:
                    structure = load_structure(file)
                    st.session_state.full_structures[file.name] = structure
                    st.success(f"Successfully loaded structure: {file.name}")

                    # Add to uploaded_files list for tracking
                    if 'uploaded_files' not in st.session_state:
                        st.session_state['uploaded_files'] = []
                    if all(f.name != file.name for f in st.session_state['uploaded_files']):
                        st.session_state['uploaded_files'].append(file)

                except Exception as e:
                    pass

def update_file_upload_section():
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'previous_uploaded_files' not in st.session_state:
        st.session_state['previous_uploaded_files'] = []

    current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
    previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

    removed_files = set(previous_file_names) - set(current_file_names)
    for removed_file in removed_files:
        if removed_file in st.session_state.full_structures:
            del st.session_state.full_structures[removed_file]
            st.success(f"Removed structure: {removed_file}")
    st.session_state['uploaded_files'] = [
        file for file in st.session_state['uploaded_files']
        if file.name in current_file_names
    ]

    handle_uploaded_files(uploaded_files_user_sidebar)


    st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []

    if st.session_state.full_structures:
        st.sidebar.subheader("📋 Currently Loaded Structures")
        for filename in st.session_state.full_structures.keys():
            st.sidebar.text(f"• {filename}")

st.sidebar.info(f"🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
st.sidebar.info(
    "Try also our XRD application **[XRDlicious](https://xrdlicious.com)**. 🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. 📺 **[Tutorial here](https://youtu.be/GGo_9T5wqus?si=xJItv-j0shr8hte_)**. Spot a bug or have a feature requests? Let us know at **lebedmi2@cvut.cz**."
    " If you like the app, please cite [**this publication**](https://doi.org/10.1016/j.jocs.2026.102846). You can consider to compile the app **locally** on your computer from **[GitHub](https://github.com/bracerino/atat-sqs-gui.git)** for better performance."
)

st.sidebar.markdown("""
<style>
.github-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    width: 100%;
    padding: 10px 16px;
    border-radius: 10px;
    border: 1px solid rgba(91, 140, 255, 0.45);
    background: linear-gradient(135deg, #57606a 0%, #6e7b8c 100%);
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
}
.github-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(91, 140, 255, 0.45);
    border-color: rgba(91, 140, 255, 0.9);
}
.github-btn svg {
    flex-shrink: 0;
    fill: #ffffff;
}
</style>
<a class="github-btn" href="https://github.com/bracerino/atat-sqs-gui.git" target="_blank">
    <svg height="20" width="20" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
        1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
        0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
        1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0
        3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0
        .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg>
    GitHub page (for local compilation)
</a>
""", unsafe_allow_html=True)
update_file_upload_section()
st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []




# Render the SQS transformation module
from more_funct.st_trans import render_sqs_module, check_sqs_mode

# Block the workflow if any uploaded structure is too large. The supercell is
# specified later in the workflow, so the uploaded structure should be a small
# unit cell rather than an already-expanded supercell.
MAX_UPLOAD_ATOMS = 1000
oversized_structures = {
    name: len(structure)
    for name, structure in st.session_state.full_structures.items()
    if len(structure) > MAX_UPLOAD_ATOMS
}

if oversized_structures:
    details = "\n".join(
        f"- **{name}**: {n_atoms} atoms" for name, n_atoms in oversized_structures.items()
    )
    st.error(
        f"⚠️ **Uploaded structure is too large (more than {MAX_UPLOAD_ATOMS} atoms).**\n\n"
        f"{details}\n\n"
        "This is **not recommended**: the supercell is specified later in the workflow, "
        "so you should upload the **smaller unit cell** instead of an already-expanded "
        "supercell. **Please remove the structure(s) listed above from the sidebar to continue.**"
    )
    st.stop()

# Call the SQS module
render_sqs_module()


st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
#def get_memory_usage():
#    process = psutil.Process(os.getpid())
#    mem_info = process.memory_info()
#    return mem_info.rss / (1024 ** 2)  # in MB

st.iframe(
    """
    <head>
        <meta name="description" content="ATAT SQS GUI: Create Input Files for Generation of SQS using ATAT mcsqs and Analyse Its Outputs">
    </head>
    """,
    height='content',
)

#memory_usage = get_memory_usage()
#st.write(
#    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. For more extensive computations, please compile the application locally from the [GitHub](https://github.com/bracerino/atat-sqs-gui.git).")
st.markdown("""
**The GUI SQS application is open-source and released under the [MIT License](https://github.com/bracerino/atat-sqs-gui/blob/main/LICENSE).**
""")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[ATAT](https://axelvandewalle.github.io/www-avdw/atat)** Licensed under the Creative Commons Attribution‑NoDerivatives 4.0 International License. **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallographic Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")


def record_and_get_pageviews():
    """Count one page view per user session and return daily view statistics.

    Views are stored in a small JSON file keyed by date. Each browser session is
    counted only once (tracked via st.session_state). Note: on Streamlit Community
    Cloud the filesystem is ephemeral, so counts reset when the app reboots.
    """
    import os
    import json
    from datetime import date

    counts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pageviews.json")
    today = date.today().isoformat()

    try:
        with open(counts_file, "r") as f:
            counts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        counts = {}

    if not st.session_state.get("_pageview_counted", False):
        counts[today] = counts.get(today, 0) + 1
        try:
            with open(counts_file, "w") as f:
                json.dump(counts, f)
            st.session_state["_pageview_counted"] = True
        except OSError:
            pass

    today_views = counts.get(today, 0)
    # Show up to the three most recent finished days (dates before today) that
    # have recorded views. If there are none yet, nothing extra is shown.
    finished = sorted(d for d in counts if d < today)[-3:]
    previous_days = [
        (f"{date.fromisoformat(d).day}.{date.fromisoformat(d).month}", counts[d])
        for d in finished
    ]
    return today_views, previous_days


try:
    today_views, previous_days = record_and_get_pageviews()
    st.sidebar.markdown("---")
    caption = f"📈 Page views today: **{today_views}**"
    if previous_days:
        caption += " (" + ", ".join(
            f"{day} - **{views}**" for day, views in previous_days) + ")"
    st.sidebar.caption(caption + ".")
except Exception:
    pass
