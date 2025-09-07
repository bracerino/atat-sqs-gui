import streamlit as st
import io
import numpy as np
import random

from helpers import *
import time
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import py3Dmol
import streamlit.components.v1 as components
import pandas as pd
from mp_api.client import MPRester
import spglib
from pymatgen.core import Structure
from aflow import search, K
from aflow import search  # ensure your file is not named aflow.py!
import aflow.keywords as AFLOW_K
import requests
import io
import re
from atat_module import *

from ase import Atoms
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
import logging
import threading
import queue
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"

if "previous_generation_mode" not in st.session_state:
    st.session_state.previous_generation_mode = "Single Run"


def create_bulk_download_zip_fixed(results, download_format, options=None):
    import zipfile
    from io import BytesIO
    import time

    if options is None:
        options = {}

    try:
        with st.spinner(f"Creating ZIP with {len(results)} {download_format} files..."):
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for result in results:
                    try:
                        file_content = generate_structure_file_content_with_options(
                            result['structure'],
                            download_format,
                            options
                        )

                        file_extension = get_file_extension(download_format)
                        filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"
                        zip_file.writestr(filename, file_content)

                    except Exception as e:
                        error_filename = f"ERROR_run_{result['run_number']}_seed_{result['seed']}.txt"
                        error_content = f"Error generating {download_format} file: {str(e)}"
                        zip_file.writestr(error_filename, error_content)

            zip_buffer.seek(0)

            timestamp = int(time.time())
            zip_filename = f"SQS_multi_run_{download_format}_{timestamp}.zip"

            st.download_button(
                label=f"📥 Download ZIP ({len(results)} files)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                type="primary",
                key=f"zip_download_{timestamp}",
                help=f"Download ZIP file containing all {len(results)} SQS structures in {download_format} format"
            )

            st.success(f"✅ ZIP file with {len(results)} {download_format} structures ready!")

    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        st.error("Please try again or check your structure files.")


def get_file_extension(file_format):
    extensions = {
        "CIF": "cif",
        "VASP": "poscar",
        "LAMMPS": "lmp",
        "XYZ": "xyz"
    }
    return extensions.get(file_format, "txt")


def generate_structure_file_content_with_options(structure, file_format, options=None):
    if options is None:
        options = {}

    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter
            cif_writer = CifWriter(structure)
            return cif_writer.__str__()

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from ase.constraints import FixAtoms
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)

            use_fractional = options.get('use_fractional', True)
            use_selective_dynamics = options.get('use_selective_dynamics', False)

            if use_selective_dynamics:
                constraint = FixAtoms(indices=[])
                ase_structure.set_constraint(constraint)

            out = StringIO()
            write(out, ase_structure, format="vasp", direct=use_fractional, sort=True)
            return out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            atom_style = options.get('atom_style', 'atomic')
            units = options.get('units', 'metal')
            include_masses = options.get('include_masses', True)
            force_skew = options.get('force_skew', False)

            out = StringIO()
            write(
                out,
                ase_structure,
                format="lammps-data",
                atom_style=atom_style,
                units=units,
                masses=include_masses,
                force_skew=force_skew
            )
            return out.getvalue()

        elif file_format == "XYZ":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="xyz")
            return out.getvalue()

        else:
            return f"Unsupported format: {file_format}"

    except Exception as e:
        return f"Error generating {file_format}: {str(e)}"


def create_bulk_download_zip(results, download_format, options=None):
    import zipfile
    from io import BytesIO

    if options is None:
        options = {}

    try:
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for result in results:
                try:
                    file_content = generate_structure_file_content_with_options(
                        result['structure'],
                        download_format,
                        options
                    )

                    file_extension = get_file_extension(download_format)
                    filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"
                    zip_file.writestr(filename, file_content)

                except Exception as e:
                    error_filename = f"ERROR_run_{result['run_number']}_seed_{result['seed']}.txt"
                    error_content = f"Error generating {download_format} file: {str(e)}"
                    zip_file.writestr(error_filename, error_content)

        zip_buffer.seek(0)

        timestamp = int(time.time())
        options_str = ""
        if download_format == "VASP" and options:
            coord_type = "frac" if options.get('use_fractional', True) else "cart"
            sd_type = "sd" if options.get('use_selective_dynamics', False) else "nosd"
            options_str = f"_{coord_type}_{sd_type}"
        elif download_format == "LAMMPS" and options:
            atom_style = options.get('atom_style', 'atomic')
            units = options.get('units', 'metal')
            options_str = f"_{atom_style}_{units}"

        zip_filename = f"SQS_multi_run_{download_format}{options_str}_{timestamp}.zip"

        zip_key = f"zip_download_bulk_{download_format}_{timestamp}"

        st.download_button(
            label=f"📥 Download {download_format} ZIP ({len(results)} files)",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            type="primary",
            key=zip_key,
            help=f"Download ZIP file containing all {len(results)} successful SQS structures in {download_format} format"
        )

        st.success(f"✅ ZIP file with {len(results)} {download_format} structures ready for download!")

    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        st.error("Please try again or contact support if the problem persists.")


def display_multi_run_results(all_results=None, download_format="CIF"):
    import numpy as np
    if all_results is None:
        if "multi_run_results" in st.session_state and st.session_state.multi_run_results:
            all_results = st.session_state.multi_run_results
        else:
            return

    if not all_results:
        return

    results_data = []
    valid_results = [r for r in all_results if r.get('best_score') is not None]

    for result in all_results:
        if result.get('best_score') is not None:
            results_data.append({
                "Run": result['run_number'],
                "Seed": result['seed'],
                "Best Score": f"{result['best_score']:.4f}",
                "Time (s)": f"{result['elapsed_time']:.1f}",
                "Atoms": len(result['structure']) if result.get('structure') else 0,
                "Status": "✅ Success"
            })
        else:
            results_data.append({
                "Run": result['run_number'],
                "Seed": result['seed'],
                "Best Score": "Failed",
                "Time (s)": f"{result.get('elapsed_time', 0):.1f}",
                "Atoms": 0,
                "Status": "❌ Error"
            })

    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_score'])
        st.success(f"🥇 **Best Result:** Run {best_result['run_number']} with score {best_result['best_score']:.4f}")

        scores = [r['best_score'] for r in valid_results]
        if len(scores) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Score", f"{min(scores):.4f}")
            with col2:
                st.metric("Worst Score", f"{max(scores):.4f}")
            with col3:
                st.metric("Average Score", f"{np.mean(scores):.4f}")
            with col4:
                st.metric("Std Dev", f"{np.std(scores):.4f}")

    st.subheader("📥 Download Individual Structures")

    col_format, col_options = st.columns([1, 2])

    with col_format:
        selected_format = st.selectbox(
            "Select format:",
            ["CIF", "VASP", "LAMMPS", "XYZ"],
            index=0,
            key="multi_run_individual_format"
        )

    format_options = {}
    with col_options:
        if selected_format == "VASP":
            st.write("**VASP Options:**")
            col_vasp1, col_vasp2 = st.columns(2)
            with col_vasp1:
                format_options['use_fractional'] = st.checkbox(
                    "Fractional coordinates",
                    value=True,
                    key="multi_vasp_fractional"
                )
            with col_vasp2:
                format_options['use_selective_dynamics'] = st.checkbox(
                    "Selective dynamics",
                    value=False,
                    key="multi_vasp_selective"
                )

        elif selected_format == "LAMMPS":
            st.write("**LAMMPS Options:**")
            col_lmp1, col_lmp2 = st.columns(2)
            with col_lmp1:
                format_options['atom_style'] = st.selectbox(
                    "Atom style:",
                    ["atomic", "charge", "full"],
                    index=0,
                    key="multi_lammps_atom_style"
                )
                format_options['units'] = st.selectbox(
                    "Units:",
                    ["metal", "real", "si"],
                    index=0,
                    key="multi_lammps_units"
                )
            with col_lmp2:
                format_options['include_masses'] = st.checkbox(
                    "Include masses",
                    value=True,
                    key="multi_lammps_masses"
                )
                format_options['force_skew'] = st.checkbox(
                    "Force triclinic",
                    value=False,
                    key="multi_lammps_skew"
                )
        else:
            st.write("")

    successful_results = [r for r in all_results if r.get('structure') is not None]

    if successful_results:
        best_run_number = min(successful_results, key=lambda x: x.get('best_score', float('inf'))).get('run_number')

        num_cols = min(4, len(successful_results))
        cols = st.columns(num_cols)

        for idx, result in enumerate(successful_results):
            with cols[idx % num_cols]:
                is_best = (result['run_number'] == best_run_number)
                button_type = "primary" if is_best else "secondary"
                label = f"📥 Run {result['run_number']}" + (" 🥇" if is_best else "")
                label += f"\nScore: {result['best_score']:.4f}"

                try:
                    file_content = generate_structure_file_content_with_options(
                        result['structure'],
                        selected_format,
                        format_options
                    )

                    file_extension = get_file_extension(selected_format)
                    filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"

                    unique_key = f"download_run_{result['run_number']}_{result['seed']}_{selected_format}_{hash(str(format_options))}"

                    st.download_button(
                        label=label,
                        data=file_content,
                        file_name=filename,
                        mime="text/plain",
                        type=button_type,
                        key=unique_key,
                        help=f"Download {selected_format} structure from run {result['run_number']}"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("No successful runs are available for download.")

    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_score'])

        st.subheader("🏆 Best Structure Analysis")
        colp1, colp2 = st.columns([1, 1])
        with colp2:
            # with st.expander(f"📊 Structure Details for Best Run {best_result['run_number']}", expanded=False):
            st.markdown(f"📊 Structure Details for Best Run {best_result['run_number']}")
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.write("**Structure Information:**")
                best_structure = best_result['structure']
                comp = best_structure.composition
                comp_data = []

                for el, amt in comp.items():
                    actual_frac = amt / comp.num_atoms
                    comp_data.append({
                        "Element": el.symbol,
                        "Count": int(amt),
                        "Fraction": f"{actual_frac:.4f}"
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True)

            with col_info2:
                st.write("**Lattice Parameters:**")
                lattice = best_structure.lattice
                st.write(f"a = {lattice.a:.4f} Å")
                st.write(f"b = {lattice.b:.4f} Å")
                st.write(f"c = {lattice.c:.4f} Å")
                st.write(f"α = {lattice.alpha:.2f}°")
                st.write(f"β = {lattice.beta:.2f}°")
                st.write(f"γ = {lattice.gamma:.2f}°")
                st.write(f"Volume = {lattice.volume:.2f} Ų")
        with colp1:
            st.write("**3D Structure Visualization:**")
            try:
                from io import StringIO
                import py3Dmol
                import streamlit.components.v1 as components
                from pymatgen.io.ase import AseAtomsAdaptor
                from ase.io import write
                import numpy as np

                jmol_colors = {
                    "H": "#FFFFFF",
                    "He": "#D9FFFF",
                    "Li": "#CC80FF",
                    "Be": "#C2FF00",
                    "B": "#FFB5B5",
                    "C": "#909090",
                    "N": "#3050F8",
                    "O": "#FF0D0D",
                    "F": "#90E050",
                    "Ne": "#B3E3F5",
                    "Na": "#AB5CF2",
                    "Mg": "#8AFF00",
                    "Al": "#BFA6A6",
                    "Si": "#F0C8A0",
                    "P": "#FF8000",
                    "S": "#FFFF30",
                    "Cl": "#1FF01F",
                    "Ar": "#80D1E3",
                    "K": "#8F40D4",
                    "Ca": "#3DFF00",
                    "Sc": "#E6E6E6",
                    "Ti": "#BFC2C7",
                    "V": "#A6A6AB",
                    "Cr": "#8A99C7",
                    "Mn": "#9C7AC7",
                    "Fe": "#E06633",
                    "Co": "#F090A0",
                    "Ni": "#50D050",
                    "Cu": "#C88033",
                    "Zn": "#7D80B0",
                    "Ga": "#C28F8F",
                    "Ge": "#668F8F",
                    "As": "#BD80E3",
                    "Se": "#FFA100",
                    "Br": "#A62929",
                    "Kr": "#5CB8D1",
                    "Rb": "#702EB0",
                    "Sr": "#00FF00",
                    "Y": "#94FFFF",
                    "Zr": "#94E0E0",
                    "Nb": "#73C2C9",
                    "Mo": "#54B5B5",
                    "Tc": "#3B9E9E",
                    "Ru": "#248F8F",
                    "Rh": "#0A7D8C",
                    "Pd": "#006985",
                    "Ag": "#C0C0C0",
                    "Cd": "#FFD98F",
                    "In": "#A67573",
                    "Sn": "#668080",
                    "Sb": "#9E63B5",
                    "Te": "#D47A00",
                    "I": "#940094",
                    "Xe": "#429EB0",
                    "Cs": "#57178F",
                    "Ba": "#00C900",
                    "La": "#70D4FF",
                    "Ce": "#FFFFC7",
                    "Pr": "#D9FFC7",
                    "Nd": "#C7FFC7",
                    "Pm": "#A3FFC7",
                    "Sm": "#8FFFC7",
                    "Eu": "#61FFC7",
                    "Gd": "#45FFC7",
                    "Tb": "#30FFC7",
                    "Dy": "#1FFFC7",
                    "Ho": "#00FF9C",
                    "Er": "#00E675",
                    "Tm": "#00D452",
                    "Yb": "#00BF38",
                    "Lu": "#00AB24",
                    "Hf": "#4DC2FF",
                    "Ta": "#4DA6FF",
                    "W": "#2194D6",
                    "Re": "#267DAB",
                    "Os": "#266696",
                    "Ir": "#175487",
                    "Pt": "#D0D0E0",
                    "Au": "#FFD123",
                    "Hg": "#B8B8D0",
                    "Tl": "#A6544D",
                    "Pb": "#575961",
                    "Bi": "#9E4FB5",
                    "Po": "#AB5C00",
                    "At": "#754F45",
                    "Rn": "#428296",
                    "Fr": "#420066",
                    "Ra": "#007D00",
                    "Ac": "#70ABFA",
                    "Th": "#00BAFF",
                    "Pa": "#00A1FF",
                    "U": "#008FFF",
                    "Np": "#0080FF",
                    "Pu": "#006BFF",
                    "Am": "#545CF2",
                    "Cm": "#785CE3",
                    "Bk": "#8A4FE3",
                    "Cf": "#A136D4",
                    "Es": "#B31FD4",
                    "Fm": "#B31FBA",
                    "Md": "#B30DA6",
                    "No": "#BD0D87",
                    "Lr": "#C70066",
                    "Rf": "#CC0059",
                    "Db": "#D1004F",
                    "Sg": "#D90045",
                    "Bh": "#E00038",
                    "Hs": "#E6002E",
                    "Mt": "#EB0026"
                }

                def add_box(view, cell, color='black', linewidth=2):
                    vertices = np.array([
                        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
                    ])
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]
                    ]
                    cart_vertices = np.dot(vertices, cell)
                    for edge in edges:
                        start, end = cart_vertices[edge[0]], cart_vertices[edge[1]]
                        view.addCylinder({
                            'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                            'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                            'radius': 0.05,
                            'color': color
                        })

                structure_ase = AseAtomsAdaptor.get_atoms(best_structure)
                xyz_io = StringIO()
                write(xyz_io, structure_ase, format="xyz")
                xyz_str = xyz_io.getvalue()

                view = py3Dmol.view(width=600, height=400)
                view.addModel(xyz_str, "xyz")
                view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})

                cell = structure_ase.get_cell()
                add_box(view, cell, color='black', linewidth=2)

                view.zoomTo()
                view.zoom(1.2)

                html_string = view._make_html()
                components.html(html_string, height=420, width=620)

                unique_elements = sorted(set(structure_ase.get_chemical_symbols()))
                legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: center; margin-top: 10px;'>"
                for elem in unique_elements:
                    color = jmol_colors.get(elem, "#CCCCCC")
                    legend_html += (
                        f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                        f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black; border-radius: 50%;'></div>"
                        f"<span style='font-weight: bold;'>{elem}</span></div>"
                    )
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error visualizing best structure: {e}")

        st.write("**PRDF Analysis:**")
        try:
            prdf_cutoff = st.session_state.get('sqs_prdf_cutoff', 10.0)
            prdf_bin_size = st.session_state.get('sqs_prdf_bin_size', 0.1)

            calculate_and_display_sqs_prdf(best_structure, cutoff=prdf_cutoff, bin_size=prdf_bin_size)

        except Exception as e:
            st.error(f"Error calculating PRDF for best structure: {e}")
            st.info("PRDF analysis could not be completed for the best structure.")

    if successful_results:
        st.subheader("📦 Bulk Download")

        col_bulk_format, col_bulk_options = st.columns([1, 2])

        with col_bulk_format:
            bulk_format = st.selectbox(
                "Bulk format:",
                ["CIF", "VASP", "LAMMPS", "XYZ"],
                index=0,
                key="bulk_format_selector"
            )

        bulk_options = {}
        with col_bulk_options:
            if bulk_format == "VASP":
                st.write("**VASP Bulk Options:**")
                col_bulk_vasp1, col_bulk_vasp2 = st.columns(2)
                with col_bulk_vasp1:
                    bulk_options['use_fractional'] = st.checkbox(
                        "Fractional coordinates",
                        value=True,
                        key="bulk_vasp_fractional"
                    )
                with col_bulk_vasp2:
                    bulk_options['use_selective_dynamics'] = st.checkbox(
                        "Selective dynamics",
                        value=False,
                        key="bulk_vasp_selective"
                    )

            elif bulk_format == "LAMMPS":
                st.write("**LAMMPS Bulk Options:**")
                col_bulk_lmp1, col_bulk_lmp2 = st.columns(2)
                with col_bulk_lmp1:
                    bulk_options['atom_style'] = st.selectbox(
                        "Atom style:",
                        ["atomic", "charge", "full"],
                        index=0,
                        key="bulk_lammps_atom_style"
                    )
                    bulk_options['units'] = st.selectbox(
                        "Units:",
                        ["metal", "real", "si"],
                        index=0,
                        key="bulk_lammps_units"
                    )
                with col_bulk_lmp2:
                    bulk_options['include_masses'] = st.checkbox(
                        "Include masses",
                        value=True,
                        key="bulk_lammps_masses"
                    )
                    bulk_options['force_skew'] = st.checkbox(
                        "Force triclinic",
                        value=False,
                        key="bulk_lammps_skew"
                    )

        if st.button("📥 Download all structures as ZIP", type="primary", key="bulk_download_button"):
            create_bulk_download_zip_fixed(successful_results, bulk_format, bulk_options)


def ase_to_pymatgen(atoms):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    lattice = Lattice(cell)
    return Structure(lattice, symbols, positions, coords_are_cartesian=True)




if "persistent_prdf_data" not in st.session_state:
    st.session_state.persistent_prdf_data = None
if "prdf_structure_key" not in st.session_state:
    st.session_state.prdf_structure_key = None


def render_sqs_module():

    st.markdown(
    """
    <h1 style='text-align: left; color: #1E3D7B;'>
        🎲 <span style='color:#2E86C1; font-weight:bold;'>SimplySQS</span>
    </h1>
    <h3 style='text-align: left; color: #444444; font-weight: normal;'>
        Generate input and analyze output files for 
        <b><span style='color:#1A7F5D;'>ATAT mcsqs</span></b> 
        to create <b><em>special quasirandom structures (SQS)</em></b>
    </h3>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    cl1, cl2,cl3 = st.columns(3)
    with cl1:
        how_cite = st.checkbox(f"📚 How to **cite**")
        if how_cite:
            with st.expander("How to cite", icon="📚", expanded=True):
                st.markdown("""
                 Please cite the following sources
                 
                 - **ATAT mcsqs method** - [VAN DE WALLE, Axel, et al. Efficient stochastic generation of special quasirandom structures. Calphad, 2013](https://www.sciencedirect.com/science/article/pii/S0364591613000540?casa_token=i1iog7eW3lQAAAAA:wxlTn-9Twj38XFx1lMfSazPb6r0JrDV7NPxeums5-2qFXHWItT2ZVu9E-IfuBjRsr7f1BEzcSw).
                 - **ATAT** - [VAN DE WALLE, Axel; ASTA, Mark; CEDER, Gerbrand. The alloy theoretic automated toolkit: A user guide. Calphad, 2002](https://www.sciencedirect.com/science/article/abs/pii/S0364591602800062).
            """)
    with cl2:
        read_more = st.checkbox(f"📖 Read **more** about **SQS**, **ATAT**, and how to **compile it**"
                               )
    if read_more:
        with st.expander("Read more", icon="📖", expanded=True):
            st.markdown("""
            ### Read More About SQS and ATAT
             Please see the following useful resources
            - [User guide how to compile ATAT mcsqs by Implant team (see also the compilation steps below)](https://implant.fs.cvut.cz/atat-mcsqs/).
            - [Tutorial explaining how to use SQS for disordered materials and generate them using ATAT mcsqs](https://cniu.me/2017/08/05/SQS.html#generate-sqs).
            - [Tutorial explaining how to generate SQS using ATAT mcsqs](https://github.com/CMSLabIITK/SQS_generation).
            - [User guide for ATAT](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual.pdf).
            - [User guide specifically for ATAT mcsqs](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/manual/node48.html).
            - [C++ code for converting bestsqs.out into POSCAR format](https://github.com/c-niu/sqs2poscar). 
            - [Python code for converting bestsqs.out into POSCAR format](https://github.com/JianboHIT/sqs2vasp).

            ### 🛠️ ATAT Installation Tutorial
            
            #### 🎥 Video Tutorial
            
            **Watch the complete compilation process:** [ATAT Compilation Video Tutorial](https://youtu.be/d5PceJoL1tw?si=akSMQ76_hKHaTKcB)
            
            #### 📝 Written Tutorial
            
            ---
            
            #### Prerequisites (Ubuntu/WSL2)
            
            First, update your system and install required packages:
            ```bash
            sudo apt-get upgrade
            sudo apt-get install tcsh
            ```
            
            #### Download and Compile ATAT
            
            1. **Download ATAT toolkit:**
            ```bash
            wget http://alum.mit.edu/www/avdw/atat/atat3_36.tar.gz
            ```
            
            2. **Extract and prepare build directory:**
            ```bash
            tar xvzf atat3_36.tar.gz
            cd atat
            mkdir build
            ```
            
            3. **Configure installation path:**
            ```bash
            nano makefile
            ```
            Change the first line from `BINDIR=...` to use the current directory path. You can get your current path with:
            ```bash
            pwd
            ```
            Then edit the makefile to set:
            ```
            BINDIR=/full/path/from/pwd/command/build
            ```
            For example, if `pwd` shows `/home/username/atat`, then set:
            ```
            BINDIR=/home/username/atat/build
            ```
            Save (Ctrl+S) and exit (Ctrl+X).
            
            4. **Compile ATAT:**
            ```bash
            make -j 10
            make install
            ```
            
            5. **Add to PATH:**
            ```bash
            cd build
            pwd  # Note this path
            nano ~/.bashrc
            ```
            Add this line at the bottom of ~/.bashrc (replace the path with your actual build directory path):
            ```bash
            export PATH=$(pwd)/build:$PATH
            ```
            Save, exit, and refresh:
            ```bash
            source ~/.bashrc
            ```
            
            6. **Test installation:**
            ```bash
            mcsqs
            ```
            This should display the list of mcsqs parameters if the compilation was successful.
        """)
    # -------------- DATABASE ----------
    with cl3:
        show_database_search = st.checkbox("🗃️ Enable **database search** (MP, AFLOW, COD)",
                                           value=False,
                                           help="🗃️ Enable to search in Materials Project, AFLOW, and COD databases")
    st.markdown("""
           <style>
           div.stButton > button[kind="primary"] {
               background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
               padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
           }
           div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
               background-color: #007acc !important; color: white !important; box-shadow: none !important;
           }

           div.stButton > button[kind="secondary"] {
               background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
               padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
           }
           div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
               background-color: #c82333 !important; color: white !important; box-shadow: none !important;
           }

           div.stButton > button[kind="tertiary"] {
               background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
               padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
           }
           div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
               background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
           }

           div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
           #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
           </style>
       """, unsafe_allow_html=True)

    def get_space_group_info(number):
        symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
        return symbol

    if show_database_search:
        with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=True):
            cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
            with cols:
                db_choices = st.multiselect(
                    "Select Database(s)",
                    options=["Materials Project", "AFLOW", "COD"],
                    default=["Materials Project", "AFLOW", "COD"],
                    help="Choose which databases to search for structures. You can select multiple databases."
                )

                if not db_choices:
                    st.warning("Please select at least one database to search.")

                st.markdown(
                    "**Maximum number of structures to be found in each database (for improving performance):**")
                col_limits = st.columns(3)

                search_limits = {}
                if "Materials Project" in db_choices:
                    with col_limits[0]:
                        search_limits["Materials Project"] = st.number_input(
                            "MP Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from Materials Project"
                        )
                if "AFLOW" in db_choices:
                    with col_limits[1]:
                        search_limits["AFLOW"] = st.number_input(
                            "AFLOW Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from AFLOW"
                        )
                if "COD" in db_choices:
                    with col_limits[2]:
                        search_limits["COD"] = st.number_input(
                            "COD Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from COD"
                        )

            with cols2:
                search_mode = st.radio(
                    "Search by:",
                    options=["Elements", "Structure ID", "Space Group + Elements", "Formula", "Search Mineral"],
                    help="Choose your search strategy"
                )

                if search_mode == "Elements":
                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )
                    search_query = " ".join(selected_elements) if selected_elements else ""

                elif search_mode == "Structure ID":
                    structure_ids = st.text_area(
                        "Enter Structure IDs (one per line):",
                        value="mp-5229\ncod_1512124\naflow:010158cb2b41a1a5",
                        help="Enter structure IDs. Examples:\n- Materials Project: mp-5229\n- COD: cod_1512124 (with cod_ prefix)\n- AFLOW: aflow:010158cb2b41a1a5 (AUID format)"
                    )

                elif search_mode == "Space Group + Elements":
                    col_sg1, col_sg2 = st.columns(2)
                    with col_sg1:
                        all_space_groups_help = "Enter space group number (1-230)\n\nAll space groups:\n\n"
                        for num in sorted(SPACE_GROUP_SYMBOLS.keys()):
                            all_space_groups_help += f"• {num}: {SPACE_GROUP_SYMBOLS[num]}\n\n"

                        space_group_number = st.number_input(
                            "Space Group Number:",
                            min_value=1,
                            max_value=230,
                            value=221,
                            help=all_space_groups_help
                        )
                        sg_symbol = get_space_group_info(space_group_number)
                        st.info(f"#:**{sg_symbol}**")

                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )

                elif search_mode == "Formula":
                    formula_input = st.text_input(
                        "Enter Chemical Formula:",
                        value="Sr Ti O3",
                        help="Enter chemical formula with spaces between elements. Examples:\n- Sr Ti O3 (strontium titanate)\n- Ca C O3 (calcium carbonate)\n- Al2 O3 (alumina)"
                    )

                elif search_mode == "Search Mineral":
                    mineral_options = []
                    mineral_mapping = {}

                    for space_group, minerals in MINERALS.items():
                        for mineral_name, formula in minerals.items():
                            option_text = f"{mineral_name} - SG #{space_group}"
                            mineral_options.append(option_text)
                            mineral_mapping[option_text] = {
                                'space_group': space_group,
                                'formula': formula,
                                'mineral_name': mineral_name
                            }

                    # Sort mineral options alphabetically
                    mineral_options.sort()

                    selected_mineral = st.selectbox(
                        "Select Mineral Structure:",
                        options=mineral_options,
                        help="Choose a mineral structure type. The exact formula and space group will be automatically set.",
                        index=2
                    )

                    if selected_mineral:
                        mineral_info = mineral_mapping[selected_mineral]

                        # col_mineral1, col_mineral2 = st.columns(2)
                        # with col_mineral1:
                        sg_symbol = get_space_group_info(mineral_info['space_group'])
                        st.info(
                            f"**Structure:** {mineral_info['mineral_name']}, **Space Group:** {mineral_info['space_group']} ({sg_symbol}), "
                            f"**Formula:** {mineral_info['formula']}")

                        space_group_number = mineral_info['space_group']
                        formula_input = mineral_info['formula']

                        st.success(
                            f"**Search will use:** Formula = {formula_input}, Space Group = {space_group_number}")

                show_element_info = st.checkbox("ℹ️ Show information about element groups")
                if show_element_info:
                    st.markdown("""
                    **Element groups note:**
                    **Common Elements (14):** H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca  
                    **Transition Metals (10):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn  
                    **Alkali Metals (6):** Li, Na, K, Rb, Cs, Fr  
                    **Alkaline Earth (6):** Be, Mg, Ca, Sr, Ba, Ra  
                    **Noble Gases (6):** He, Ne, Ar, Kr, Xe, Rn  
                    **Halogens (5):** F, Cl, Br, I, At  
                    **Lanthanides (15):** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu  
                    **Actinides (15):** Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr  
                    **Other Elements (51):** All remaining elements
                    """)

            if st.button("Search Selected Databases"):
                if not db_choices:
                    st.error("Please select at least one database to search.")
                else:
                    for db_choice in db_choices:
                        if db_choice == "Materials Project":
                            mp_limit = search_limits.get("Materials Project", 50)
                            with st.spinner(f"Searching **the MP database** (limit: {mp_limit}), please wait. 😊"):
                                try:
                                    with MPRester(MP_API_KEY) as mpr:
                                        docs = None

                                        if search_mode == "Elements":
                                            elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                            if not elements_list:
                                                st.error("Please enter at least one element for the search.")
                                                continue
                                            elements_list_sorted = sorted(set(elements_list))
                                            docs = mpr.materials.summary.search(
                                                elements=elements_list_sorted,
                                                num_elements=len(elements_list_sorted),
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Structure ID":
                                            mp_ids = [id.strip() for id in structure_ids.split('\n')
                                                      if id.strip() and id.strip().startswith('mp-')]
                                            if not mp_ids:
                                                st.warning(
                                                    "No valid Materials Project IDs found (should start with 'mp-')")
                                                continue
                                            docs = mpr.materials.summary.search(
                                                material_ids=mp_ids,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Space Group + Elements":
                                            elements_list = sorted(set(selected_elements))
                                            if not elements_list:
                                                st.warning(
                                                    "Please select elements for Materials Project space group search.")
                                                continue

                                            search_params = {
                                                "elements": elements_list,
                                                "num_elements": len(elements_list),
                                                "fields": ["material_id", "formula_pretty", "symmetry", "nsites",
                                                           "volume"],
                                                "spacegroup_number": space_group_number
                                            }

                                            docs = mpr.materials.summary.search(**search_params)

                                        elif search_mode == "Formula":
                                            if not formula_input.strip():
                                                st.warning(
                                                    "Please enter a chemical formula for Materials Project search.")
                                                continue

                                            # Convert space-separated format to compact format (Sr Ti O3 -> SrTiO3)
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Search Mineral":
                                            if not selected_mineral:
                                                st.warning(
                                                    "Please select a mineral structure for Materials Project search.")
                                                continue
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            # Search by formula and space group
                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                spacegroup_number=space_group_number,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        if docs:
                                            status_placeholder = st.empty()
                                            st.session_state.mp_options = []
                                            st.session_state.full_structures_see = {}
                                            limited_docs = docs[:mp_limit]

                                            for doc in limited_docs:
                                                full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                                  conventional_unit_cell=True)
                                                st.session_state.full_structures_see[doc.material_id] = full_structure
                                                lattice = full_structure.lattice
                                                leng = len(full_structure)
                                                lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                               f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                                st.session_state.mp_options.append(
                                                    f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}) [{lattice_str}], {float(doc.volume):.1f} Å³, {leng} atoms"
                                                )
                                                status_placeholder.markdown(
                                                    f"- **Structure loaded:** `{full_structure.composition.reduced_formula}` ({doc.material_id})"
                                                )
                                            if len(limited_docs) < len(docs):
                                                st.info(
                                                    f"Showing first {mp_limit} of {len(docs)} total Materials Project results. Increase limit to see more.")
                                            st.success(
                                                f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                        else:
                                            st.session_state.mp_options = []
                                            st.warning("No matching structures found in Materials Project.")
                                except Exception as e:
                                    st.error(f"An error occurred with Materials Project: {e}")

                        elif db_choice == "AFLOW":
                            aflow_limit = search_limits.get("AFLOW", 50)
                            with st.spinner(f"Searching **the AFLOW database** (limit: {aflow_limit}), please wait. 😊"):
                                try:
                                    results = []

                                    if search_mode == "Elements":
                                        elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                        if not elements_list:
                                            st.warning("Please enter elements for AFLOW search.")
                                            continue
                                        ordered_elements = sorted(elements_list)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        results = list(
                                            search(catalog="icsd")
                                            .filter(
                                                (AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                                            .select(
                                                AFLOW_K.auid,
                                                AFLOW_K.compound,
                                                AFLOW_K.geometry,
                                                AFLOW_K.spacegroup_relax,
                                                AFLOW_K.aurl,
                                                AFLOW_K.files,
                                            )
                                        )

                                    elif search_mode == "Structure ID":
                                        aflow_auids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('aflow:'):
                                                auid = id_line.replace('aflow:', '').strip()
                                                aflow_auids.append(auid)

                                        if not aflow_auids:
                                            st.warning("No valid AFLOW AUIDs found (should start with 'aflow:')")
                                            continue

                                        results = []
                                        for auid in aflow_auids:
                                            try:
                                                result = list(search(catalog="icsd")
                                                              .filter(AFLOW_K.auid == f"aflow:{auid}")
                                                              .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                      AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                      AFLOW_K.files))
                                                results.extend(result)
                                            except Exception as e:
                                                st.warning(f"AFLOW search failed for AUID '{auid}': {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        if not selected_elements:
                                            st.warning("Please select elements for AFLOW space group search.")
                                            continue
                                        ordered_elements = sorted(selected_elements)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        try:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.species % ordered_str) &
                                                                   (AFLOW_K.nspecies == aflow_nspecies) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))
                                        except Exception as e:
                                            st.warning(f"AFLOW space group search failed: {e}")
                                            results = []


                                    elif search_mode == "Formula":

                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for AFLOW search.")

                                            continue

                                        def convert_to_aflow_formula(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Add "1" if no number

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        # Generate 2x multiplied formula
                                        def multiply_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = int(match.group(2)) if match.group(2) else 1

                                                    elements_dict[element] = str(count * 2)  # Multiply by 2

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula(formula_input)

                                        aflow_formula_2x = multiply_formula_by_2(formula_input)

                                        if aflow_formula_2x != aflow_formula:

                                            results = list(search(catalog="icsd")

                                                           .filter((AFLOW_K.compound == aflow_formula) |

                                                                   (AFLOW_K.compound == aflow_formula_2x))

                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,

                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching for both {aflow_formula} and {aflow_formula_2x} formulas simultaneously")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter(AFLOW_K.compound == aflow_formula)
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(f"Searching for formula {aflow_formula}")


                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for AFLOW search.")
                                            continue

                                        def convert_to_aflow_formula_mineral(formula_input):
                                            import re
                                            formula_parts = formula_input.strip().split()
                                            elements_dict = {}
                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Always add "1" for single atoms

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        def multiply_mineral_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:
                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)
                                                    count = int(match.group(2)) if match.group(2) else 1
                                                    elements_dict[element] = str(count * 2)  # Multiply by 2
                                            aflow_parts = []
                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")
                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula_mineral(formula_input)

                                        aflow_formula_2x = multiply_mineral_formula_by_2(formula_input)

                                        # Search for both formulas with space group constraint in a single query

                                        if aflow_formula_2x != aflow_formula:
                                            results = list(search(catalog="icsd")
                                                           .filter(((AFLOW_K.compound == aflow_formula) |
                                                                    (AFLOW_K.compound == aflow_formula_2x)) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for both {aflow_formula} and {aflow_formula_2x} with space group {space_group_number}")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.compound == aflow_formula) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for formula {aflow_formula} with space group {space_group_number}")

                                    if results:
                                        status_placeholder = st.empty()
                                        st.session_state.aflow_options = []
                                        st.session_state.entrys = {}

                                        limited_results = results[:aflow_limit]

                                        for entry in limited_results:
                                            st.session_state.entrys[entry.auid] = entry
                                            st.session_state.aflow_options.append(
                                                f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax}) {entry.geometry}"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                            )
                                        if len(limited_results) < len(results):
                                            st.info(
                                                f"Showing first {aflow_limit} of {len(results)} total AFLOW results. Increase limit to see more.")
                                        st.success(f"Found {len(st.session_state.aflow_options)} structures in AFLOW.")
                                    else:
                                        st.session_state.aflow_options = []
                                        st.warning("No matching structures found in AFLOW.")
                                except Exception as e:
                                    st.warning(f"No matching structures found in AFLOW.")
                                    st.session_state.aflow_options = []

                        elif db_choice == "COD":
                            cod_limit = search_limits.get("COD", 50)
                            with st.spinner(f"Searching **the COD database** (limit: {cod_limit}), please wait. 😊"):
                                try:
                                    cod_entries = []

                                    if search_mode == "Elements":
                                        elements = [el.strip() for el in search_query.split() if el.strip()]
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please enter elements for COD search.")
                                            continue

                                    elif search_mode == "Structure ID":
                                        cod_ids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('cod_'):
                                                # Extract numeric ID from cod_XXXXX format
                                                numeric_id = id_line.replace('cod_', '').strip()
                                                if numeric_id.isdigit():
                                                    cod_ids.append(numeric_id)

                                        if not cod_ids:
                                            st.warning(
                                                "No valid COD IDs found (should start with 'cod_' followed by numbers)")
                                            continue

                                        cod_entries = []
                                        for cod_id in cod_ids:
                                            try:
                                                params = {'format': 'json', 'detail': '1', 'id': cod_id}
                                                entry = get_cod_entries(params)
                                                if entry:
                                                    if isinstance(entry, list):
                                                        cod_entries.extend(entry)
                                                    else:
                                                        cod_entries.append(entry)
                                            except Exception as e:
                                                st.warning(f"COD search failed for ID {cod_id}: {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        elements = selected_elements
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            params['space_group_number'] = str(space_group_number)

                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please select elements for COD space group search.")
                                            continue

                                    elif search_mode == "Formula":
                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for COD search.")
                                            continue

                                        # alphabet sorting
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        print(alphabet_form)
                                        params = {'format': 'json', 'detail': '1', 'formula': alphabet_form}
                                        cod_entries = get_cod_entries(params)

                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for COD search.")
                                            continue

                                        # Use both formula and space group for COD search
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        params = {
                                            'format': 'json',
                                            'detail': '1',
                                            'formula': alphabet_form,
                                            'space_group_number': str(space_group_number)
                                        }
                                        cod_entries = get_cod_entries(params)

                                    if cod_entries and isinstance(cod_entries, list):
                                        st.session_state.cod_options = []
                                        st.session_state.full_structures_see_cod = {}
                                        status_placeholder = st.empty()
                                        limited_entries = cod_entries[:cod_limit]
                                        errors = []

                                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                            future_to_entry = {executor.submit(fetch_and_parse_cod_cif, entry): entry
                                                               for
                                                               entry in limited_entries}

                                            processed_count = 0
                                            for future in concurrent.futures.as_completed(future_to_entry):
                                                processed_count += 1
                                                status_placeholder.markdown(
                                                    f"- **Processing:** {processed_count}/{len(limited_entries)} entries...")
                                                try:
                                                    cod_id, structure, entry_data, error = future.result()
                                                    if error:
                                                        original_entry = future_to_entry[future]
                                                        errors.append(
                                                            f"Entry `{original_entry.get('file', 'N/A')}` failed: {error}")
                                                        continue  # Skip to the next completed future
                                                    if cod_id and structure and entry_data:
                                                        st.session_state.full_structures_see_cod[cod_id] = structure

                                                        spcs = entry_data.get("sg", "Unknown")
                                                        spcs_number = entry_data.get("sgNumber", "Unknown")
                                                        cell_volume = structure.lattice.volume
                                                        option_str = (
                                                            f"{cod_id}: {structure.composition.reduced_formula} ({spcs} #{spcs_number}) [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f}, "
                                                            f"{structure.lattice.beta:.2f}, {structure.lattice.gamma:.2f}°], {cell_volume:.1f} Å³, {len(structure)} atoms"
                                                        )
                                                        st.session_state.cod_options.append(option_str)

                                                except Exception as e:
                                                    errors.append(
                                                        f"A critical error occurred while processing a result: {e}")
                                        status_placeholder.empty()
                                        if st.session_state.cod_options:
                                            if len(limited_entries) < len(cod_entries):
                                                st.info(
                                                    f"Showing first {cod_limit} of {len(cod_entries)} total COD results. Increase limit to see more.")
                                            st.success(
                                                f"Found and processed {len(st.session_state.cod_options)} structures from COD.")
                                        else:
                                            st.warning("COD: No matching structures could be successfully processed.")
                                        if errors:
                                            st.error(f"Encountered {len(errors)} error(s) during the search.")
                                            with st.container(border=True):
                                                for e in errors:
                                                    st.warning(e)
                                    else:
                                        st.session_state.cod_options = []
                                        st.warning("COD: No matching structures found.")
                                except Exception as e:
                                    st.warning(f"COD search error: {e}")
                                    st.session_state.cod_options = []

            # with cols2:
            #     image = Image.open("images/Rabbit2.png")
            #     st.image(image, use_container_width=True)

            with cols3:
                if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options']):
                    tabs = []
                    if 'mp_options' in st.session_state and st.session_state.mp_options:
                        tabs.append("Materials Project")
                    if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                        tabs.append("AFLOW")
                    if 'cod_options' in st.session_state and st.session_state.cod_options:
                        tabs.append("COD")

                    if tabs:
                        selected_tab = st.tabs(tabs)

                        tab_index = 0
                        if 'mp_options' in st.session_state and st.session_state.mp_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in Materials Project")
                                selected_structure = st.selectbox("Select a structure from MP:",
                                                                  st.session_state.mp_options)
                                selected_id = selected_structure.split(":")[0].strip()
                                composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
                                file_name = f"{selected_id}_{composition}.cif"
                                file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                                if selected_id in st.session_state.full_structures_see:
                                    selected_entry = st.session_state.full_structures_see[selected_id]

                                    conv_lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**Material ID:** {selected_id}, **Formula:** {composition}, N. of Atoms {n_atoms}")

                                    st.write(
                                        f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    mp_url = f"https://materialsproject.org/materials/{selected_id}"
                                    st.write(f"**Link:** {mp_url}")

                                    col_mpd, col_mpb = st.columns([2, 1])
                                    with col_mpd:
                                        if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                            pmg_structure = st.session_state.full_structures_see[selected_id]
                                            # check_structure_size_and_warn(pmg_structure, f"MP structure {selected_id}")
                                            st.session_state.full_structures[file_name] = pmg_structure
                                            cif_writer = CifWriter(pmg_structure)
                                            cif_content = cif_writer.__str__()
                                            cif_file = io.BytesIO(cif_content.encode('utf-8'))
                                            cif_file.name = file_name
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            if all(f.name != file_name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from Materials Project!")
                                    with col_mpb:
                                        st.download_button(
                                            label="Download MP CIF",
                                            data=str(
                                                CifWriter(st.session_state.full_structures_see[selected_id],
                                                          symprec=0.01)),
                                            file_name=file_name,
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                st.info(
                                    f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                            tab_index += 1

                        if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in AFLOW")
                                st.warning(
                                    "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                                selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                                  st.session_state.aflow_options)
                                selected_auid = selected_structure.split(": ")[0].strip()
                                selected_entry = next(
                                    (entry for entry in st.session_state.entrys.values() if
                                     entry.auid == selected_auid),
                                    None)
                                if selected_entry:

                                    cif_files = [f for f in selected_entry.files if
                                                 f.endswith("_sprim.cif") or f.endswith(".cif")]

                                    if cif_files:

                                        cif_filename = cif_files[0]

                                        # Correct the AURL: replace the first ':' with '/'

                                        host_part, path_part = selected_entry.aurl.split(":", 1)

                                        corrected_aurl = f"{host_part}/{path_part}"

                                        file_url = f"http://{corrected_aurl}/{cif_filename}"
                                        response = requests.get(file_url)
                                        cif_content = response.content

                                        structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'),
                                                                                  fmt="cif")
                                        converted_structure = get_full_conventional_structure(structure_from_aflow,
                                                                                              symprec=0.1)

                                        conv_lattice = converted_structure.lattice
                                        cell_volume = converted_structure.lattice.volume
                                        density = str(converted_structure.density).split()[0]
                                        n_atoms = len(converted_structure)
                                        atomic_den = n_atoms / cell_volume

                                        structure_type = identify_structure_type(converted_structure)
                                        st.write(f"**Structure type:** {structure_type}")
                                        analyzer = SpacegroupAnalyzer(structure_from_aflow)
                                        st.write(
                                            f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
                                        st.write(
                                            f"**AUID:** {selected_entry.auid}, **Formula:** {selected_entry.compound}, **N. of Atoms:** {n_atoms}")
                                        st.write(
                                            f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, "
                                            f"γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                        st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                        linnk = f"https://aflowlib.duke.edu/search/ui/material/?id=" + selected_entry.auid
                                        st.write("**Link:**", linnk)

                                        if st.button("Add Selected Structure (AFLOW)", key="add_btn_aflow"):
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            cif_file = io.BytesIO(cif_content)
                                            cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                                            st.session_state.full_structures[cif_file.name] = structure_from_aflow

                                            # check_structure_size_and_warn(structure_from_aflow, cif_file.name)
                                            if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from AFLOW!")

                                        st.download_button(
                                            label="Download AFLOW CIF",
                                            data=cif_content,
                                            file_name=f"{selected_entry.compound}_{selected_entry.auid}.cif",
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                        st.info(
                                            f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                                    else:
                                        st.warning("No CIF file found for this AFLOW entry.")
                            tab_index += 1

                        # COD tab
                        if 'cod_options' in st.session_state and st.session_state.cod_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in COD")
                                selected_cod_structure = st.selectbox(
                                    "Select a structure from COD:",
                                    st.session_state.cod_options,
                                    key='sidebar_select_cod'
                                )
                                cod_id = selected_cod_structure.split(":")[0].strip()
                                if cod_id in st.session_state.full_structures_see_cod:
                                    selected_entry = st.session_state.full_structures_see_cod[cod_id]
                                    lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    idcodd = cod_id.removeprefix("cod_")

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**COD ID:** {idcodd}, **Formula:** {selected_entry.composition.reduced_formula}, **N. of Atoms:** {n_atoms}")
                                    st.write(
                                        f"**Conventional Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                                    st.write(f"**Link:** {cod_url}")

                                    file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                                    if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                                        cif_writer = CifWriter(selected_entry, symprec=0.01)
                                        cif_data = str(cif_writer)
                                        st.session_state.full_structures[file_name] = selected_entry
                                        cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                        cif_file.name = file_name
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)

                                        # check_structure_size_and_warn(selected_entry, file_name)
                                        st.success("Structure added from COD!")

                                    st.download_button(
                                        label="Download COD CIF",
                                        data=str(CifWriter(selected_entry, symprec=0.01)),
                                        file_name=file_name,
                                        mime="chemical/x-cif", type="primary",
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")


    if "sqs_mode_initialized" not in st.session_state:
        if "calc_xrd" not in st.session_state:
            st.session_state.calc_xrd = False
        st.session_state.sqs_mode_initialized = True

    if 'full_structures' in st.session_state and st.session_state['full_structures']:
        file_options = list(st.session_state['full_structures'].keys())

        selected_sqs_file = st.sidebar.selectbox(
            "Select structure for SQS transformation:",
            file_options,
            key="sqs_structure_selector"
        )

        try:
                # Add the ATAT section here
            render_atat_sqs_section()

        except Exception as e:
            st.error(f"Error loading structure: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("""
            Special Quasirandom Structures (SQS) approximate random alloys by matching the correlation functions 
            of a truly random alloy in a finite supercell.
        """)
        intro_text()


def check_sqs_mode(calc_mode):
    if "previous_calc_mode" not in st.session_state:
        st.session_state.previous_calc_mode = calc_mode.copy()
    if "🎲 SQS Transformation" in calc_mode and "🎲 SQS Transformation" not in st.session_state.previous_calc_mode:
        st.cache_data.clear()
        st.cache_resource.clear()
        calc_mode = ["🎲 SQS Transformation"]
        if "sqs_mode_initialized" in st.session_state:
            del st.session_state.sqs_mode_initialized
        # st.rerun()

    if "🎲 SQS Transformation" in calc_mode and len(calc_mode) > 1:
        calc_mode = ["🎲 SQS Transformation"]
        # st.rerun()

    st.session_state.previous_calc_mode = calc_mode.copy()
    return calc_mode
