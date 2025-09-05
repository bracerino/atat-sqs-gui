import streamlit as st
import pandas as pd
from pymatgen.core import Structure
import concurrent.futures
import requests
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from math import cos, radians, sqrt
import io
import re
import spglib

def calculate_achievable_concentrations(target_concentrations, supercell_multiplier):
    if not isinstance(supercell_multiplier, int) or supercell_multiplier <= 0:
        raise ValueError("supercell_multiplier must be a positive integer.")
    if not target_concentrations:
        return {}, {}

    total_frac = sum(target_concentrations.values())
    if abs(total_frac - 1.0) > 1e-6:
        normalized_targets = {el: frac / total_frac for el, frac in target_concentrations.items()}
    else:
        normalized_targets = target_concentrations

    quotas = {el: frac * supercell_multiplier for el, frac in normalized_targets.items()}
    achievable_counts = {el: int(q) for el, q in quotas.items()}
    remaining_atoms = supercell_multiplier - sum(achievable_counts.values())
    remainders = {el: quotas[el] - achievable_counts[el] for el in quotas}
    sorted_by_remainder = sorted(remainders.keys(), key=lambda el: remainders[el], reverse=True)

    for i in range(remaining_atoms):
        element_to_increment = sorted_by_remainder[i]
        achievable_counts[element_to_increment] += 1

    if sum(achievable_counts.values()) != supercell_multiplier:
        raise RuntimeError("Fatal error in apportionment logic.")

    achievable_concentrations = {
        el: count / supercell_multiplier for el, count in achievable_counts.items()
    }
    return achievable_concentrations, achievable_counts


def intro_text():
    st.warning(f"Please upload a **crystal structure** file (or search for the structure with the implemented interface within **MP, AFLOW, or "
               "COD databases**) to use the SQS tool.")

    st.markdown("""
---
### 🎥 Tutorials
- 📘 **General Overview of the App**: [Watch on YouTube](https://www.youtube.com/watch?v=GGo_9T5wqus)  
- ⚡ **Binary Alloy Workflow (Search Across Concentration Range)**: [Watch on YouTube](https://youtu.be/wL5re3Fu1nQ?si=4HGRmIQBX39zs-0B)

---

This tool provides a **graphical interface** for generating input files  
(`rndstr.in`, `sqscell.out`) to create **Special Quasi-Random Structures (SQS)** using the **ATAT mcsqs** package.  

It also enables **automated script generation** for binary alloys and batch execution of mcsqs searches across composition ranges.  

---

### ✨ Key Features

#### 🔬 Crystal Structures
- Upload custom crystal structures  
- Retrieve directly from **Materials Project (MP)**, **AFLOW**, or **COD** databases  

---

#### 🎯 Supercell & Concentrations
- Define **supercell size / number of atoms**
- Set target **sublattice concentrations**  
- 📊 Automatically recalculate feasible concentrations  

---

#### 💾 Input File Generation
- Generate **`rndstr.in`** and **`sqscell.out`** files instantly  

---

#### 🛠️ Automated Workflows
- Create **bash scripts** to:  
  - Build input files  
  - Run **mcsqs** with optional parallelization  
  - Monitor convergence in real-time  
- Automate full SQS searches **across binary alloy composition ranges**  

---

#### 📤 & 📥 File Handling
- Convert **`bestsqs.out`** into:  
  - POSCAR | LMP | CIF | XYZ  
- Upload log files (`mcsqs.log`, `mcsqs1.log`, …, `mcsqs_progress.csv`) for:  
  - Convergence analysis  
  - Best run detection  

---

#### 📈 Structure Analysis
- Compute the **Pair Radial Distribution Function (PRDF)** for `bestsqs.out`  

---

#### 🧹 Vacancy Engineering
- Introduce **ordered vacancies** by selectively removing elements from SQS  

---

🚀 This tool streamlines the **entire SQS workflow** — from structure input to automated search, analysis, and output conversion.
     
     """)



import streamlit.components.v1 as components
import numpy as np
import py3Dmol
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write


def structure_preview(working_structure):
    #st.subheader("Structure Preview")

    lattice = working_structure.lattice
    st.write(f"**Lattice parameters:**")
    st.write(f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å")
    st.write(f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")

    st.write("**Structure visualization:**")

    try:
        from io import StringIO

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

        structure_ase = AseAtomsAdaptor.get_atoms(working_structure)
        xyz_io = StringIO()
        write(xyz_io, structure_ase, format="xyz")
        xyz_str = xyz_io.getvalue()

        view = py3Dmol.view(width=400, height=400)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.4, "colorscheme": "Jmol"}})

        cell = structure_ase.get_cell()
        add_box(view, cell, color='black', linewidth=2)

        view.zoomTo()
        view.zoom(1.2)

        html_string = view._make_html()
        components.html(html_string, height=420, width=420)

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
        st.error(f"Error visualizing structure: {e}")
        st.info("3D visualization is not available, but you can still generate the SQS structure.")


def sqs_visualization(result):
    try:
        from io import StringIO
        import numpy as np

        jmol_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
            'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
            'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
            'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
            'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0',
            'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929',
            'Kr': '#5CB8D1', 'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
            'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F', 'Rh': '#0A7D8C',
            'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F', 'In': '#A67573', 'Sn': '#668080',
            'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094', 'Xe': '#429EB0', 'Cs': '#57178F',
            'Ba': '#00C900', 'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
            'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7', 'Tb': '#30FFC7',
            'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675', 'Tm': '#00D452', 'Yb': '#00BF38',
            'Lu': '#00AB24', 'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB',
            'Os': '#266696', 'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
            'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00', 'At': '#754F45',
            'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00', 'Ac': '#70ABFA', 'Th': '#00BAFF',
            'Pa': '#00A1FF', 'U': '#008FFF', 'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2',
            'Cm': '#785CE3', 'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
            'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066', 'Rf': '#CC0059', 'Db': '#D1004F',
            'Sg': '#D90045', 'Bh': '#E00038', 'Hs': '#E6002E', 'Mt': '#EB0026'
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

        show_lattice_vectors = True
        show_unit_cell = True

        structure_ase = AseAtomsAdaptor.get_atoms(result['structure'])
        xyz_io = StringIO()
        write(xyz_io, structure_ase, format="xyz")
        xyz_str = xyz_io.getvalue()

        view = py3Dmol.view(width=600, height=400)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})

        cell_3dmol = structure_ase.get_cell()

        if show_unit_cell and np.linalg.det(cell_3dmol) > 1e-6:
            add_box(view, cell_3dmol, color='black', linewidth=2)
        if show_lattice_vectors and np.linalg.det(cell_3dmol) > 1e-6:
            a, b, c = cell_3dmol[0], cell_3dmol[1], cell_3dmol[2]

            view.addArrow({
                'start': {'x': 0, 'y': 0, 'z': 0},
                'end': {'x': a[0], 'y': a[1], 'z': a[2]},
                'color': 'red',
                'radius': 0.1
            })
            view.addArrow({
                'start': {'x': 0, 'y': 0, 'z': 0},
                'end': {'x': b[0], 'y': b[1], 'z': b[2]},
                'color': 'green',
                'radius': 0.1
            })
            view.addArrow({
                'start': {'x': 0, 'y': 0, 'z': 0},
                'end': {'x': c[0], 'y': c[1], 'z': c[2]},
                'color': 'blue',
                'radius': 0.1
            })


            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            c_norm = np.linalg.norm(c)

            view.addLabel(f"a = {a_norm:.3f} Å", {
                "position": {"x": a[0] * 1.1, "y": a[1] * 1.1, "z": a[2] * 1.1},
                "backgroundColor": "red",
                "fontColor": "white",
                "fontSize": 12
            })
            view.addLabel(f"b = {b_norm:.3f} Å", {
                "position": {"x": b[0] * 1.1, "y": b[1] * 1.1, "z": b[2] * 1.1},
                "backgroundColor": "green",
                "fontColor": "white",
                "fontSize": 12
            })
            view.addLabel(f"c = {c_norm:.3f} Å", {
                "position": {"x": c[0] * 1.1, "y": c[1] * 1.1, "z": c[2] * 1.1},
                "backgroundColor": "blue",
                "fontColor": "white",
                "fontSize": 12
            })

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
        st.error(f"Error visualizing SQS structure: {e}")
        import traceback
        st.error(f"Debug: {traceback.format_exc()}")


def generated_SQS_information(result):
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.write("**Final Composition:**")
        comp = result['structure'].composition
        comp_data = []

        target_concentrations = result['target_concentrations']
        achievable_concentrations = result['achievable_concentrations']

        if isinstance(target_concentrations, dict) and any(isinstance(v, dict) for v in target_concentrations.values()):
            global_target = {}
            global_achievable = {}

            for sublattice_id, sublattice_conc in target_concentrations.items():
                for element, conc in sublattice_conc.items():
                    if element in global_target:
                        global_target[element] += conc
                    else:
                        global_target[element] = conc

            for sublattice_id, sublattice_conc in achievable_concentrations.items():
                for element, conc in sublattice_conc.items():
                    if element in global_achievable:
                        global_achievable[element] += conc
                    else:
                        global_achievable[element] = conc

            num_sublattices = len(target_concentrations)
            for element in global_target:
                global_target[element] /= num_sublattices
            for element in global_achievable:
                global_achievable[element] /= num_sublattices

        else:
            global_target = target_concentrations
            global_achievable = achievable_concentrations

        for el, amt in comp.items():
            target_frac = global_target.get(el.symbol, 0.0)
            achievable_frac = global_achievable.get(el.symbol, 0.0)
            actual_frac = amt / comp.num_atoms

            comp_data.append({
                "Element": el.symbol,
                "Count": int(amt),
                "Actual": f"{actual_frac:.4f}",
                "Target": f"{target_frac:.4f}",
                "Match": "✅" if abs(actual_frac - target_frac) < 0.01 else "⚠️"
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

    with col_info2:
        st.write("**Lattice Parameters:**")
        sqs_lattice = result['structure'].lattice
        st.write(f"a = {sqs_lattice.a:.4f} Å")
        st.write(f"b = {sqs_lattice.b:.4f} Å")
        st.write(f"c = {sqs_lattice.c:.4f} Å")
        st.write(f"α = {sqs_lattice.alpha:.2f}°")
        st.write(f"β = {sqs_lattice.beta:.2f}°")
        st.write(f"γ = {sqs_lattice.gamma:.2f}°")
        st.write(f"Volume = {sqs_lattice.volume:.2f} Ų")

    if isinstance(target_concentrations, dict) and any(isinstance(v, dict) for v in target_concentrations.values()):
        with st.expander("🎯 Sublattice-Specific Composition Details", expanded=False):
            st.write("**Sublattice Breakdown:**")
            sublattice_data = []

            for sublattice_id in sorted(target_concentrations.keys()):
                target_sub = target_concentrations[sublattice_id]
                achievable_sub = achievable_concentrations.get(sublattice_id, {})

                for element in target_sub:
                    target_val = target_sub[element]
                    achievable_val = achievable_sub.get(element, 0.0)
                    status = "✅" if abs(target_val - achievable_val) < 0.01 else "⚠️"

                    sublattice_data.append({
                        "Sublattice": sublattice_id,
                        "Element": element,
                        "Target": f"{target_val:.4f}",
                        "Achievable": f"{achievable_val:.4f}",
                        "Status": status
                    })

            if sublattice_data:
                sublattice_df = pd.DataFrame(sublattice_data)
                st.dataframe(sublattice_df, use_container_width=True)



def has_partial_occupancies(structure):
    for site in structure:
        if not site.is_ordered:
            return True
    return False


def get_sublattice_composition_options():
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Vac'
    ]
    return common_elements


import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def remove_vacancies_from_structure(structure):
    non_vacancy_indices = []
    for i, site in enumerate(structure):
        if site.is_ordered:
            if site.specie.symbol != 'X':
                non_vacancy_indices.append(i)
        else:
            has_non_vacancy = any(sp.symbol != 'X' for sp in site.species)
            if has_non_vacancy:
                non_vacancy_indices.append(i)

    if len(non_vacancy_indices) == len(structure):
        return structure

    new_lattice = structure.lattice
    new_species = []
    new_coords = []

    for i in non_vacancy_indices:
        site = structure[i]
        new_coords.append(site.frac_coords)
        if site.is_ordered:
            new_species.append(site.specie)
        else:

            filtered_species = {sp: occ for sp, occ in site.species.items() if sp.symbol != 'X'}

            total_occ = sum(filtered_species.values())
            if total_occ > 0:
                normalized_species = {sp: occ / total_occ for sp, occ in filtered_species.items()}
                new_species.append(normalized_species)
            else:

                continue

    new_structure = Structure(new_lattice, new_species, new_coords)
    return new_structure


def render_site_sublattice_selector(working_structure, all_sites):
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )
    st.subheader("4️⃣ Step 4: Create Sublattices - Select Elements and Concentrations")
    st.info(
        "Select atomic sites and specify elements with concentrations. Sites with identical compositions will form sublattices.")

    if "site_assignments" not in st.session_state:
        st.session_state.site_assignments = {}

    with st.expander("📋 All Atomic Sites", expanded=False):
        site_data = []
        for site_info in all_sites:
            site_data.append({
                "Site Index": site_info['site_index'],
                "Current Element": site_info['element'],
                "Wyckoff Letter": site_info['wyckoff_letter'],
                "Coordinates": f"({site_info['coords'][0]:.3f}, {site_info['coords'][1]:.3f}, {site_info['coords'][2]:.3f})"
            })
        site_df = pd.DataFrame(site_data)
        st.dataframe(site_df, use_container_width=True)

    assigned_sites = set()
    for assignment_key in st.session_state.site_assignments.keys():
        assigned_sites.update(assignment_key)

    unassigned_sites = [site['site_index'] for site in all_sites if site['site_index'] not in assigned_sites]

    st.write(
        "**Assign Elements to Sites. Then confirm it with the 'Set elements' button. After that, you can assign another sites differently if needed:**")

    if unassigned_sites:
        site_options = unassigned_sites
        site_labels = [f"Site {site['site_index']}: {site['element']} (Wyckoff {site['wyckoff_letter']})"
                       for site in all_sites if site['site_index'] in unassigned_sites]

        selected_sites = st.multiselect(
            f"Select from {len(unassigned_sites)} unassigned sites:",
            options=site_options,
            format_func=lambda x: site_labels[unassigned_sites.index(x)],
            key="selected_sites"
        )

        if selected_sites:
            st.write(f"**Configure elements for sites: {selected_sites}**")

            common_elements = [
                'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Cs', 'Ba', 'La', 'Ce', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'U'
            ]
            all_elements = ['X'] + sorted(common_elements)

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_elements = st.multiselect(
                    "Select elements (use 'X' for vacancy):",
                    options=all_elements,
                    default=[],
                    key="elements_multiselect",
                    help="Example: Select 'Fe' and 'Ga' or 'Ti' and 'Al'"
                )

            with col2:
                if st.button("Set Elements", key="set_elements_btn", type='secondary'):
                    if selected_elements:

                        elements_list = sorted(selected_elements)

                        assignment_key = tuple(sorted(selected_sites))
                        st.session_state.site_assignments[assignment_key] = {
                            'elements': elements_list,
                            'concentrations': {}
                        }
                        st.success(
                            f"Elements {elements_list} assigned to sites {selected_sites} (sorted alphabetically)")

                        st.rerun()
                    else:
                        st.warning("Please select at least one element.")
    else:
        st.info("✅ All sites have been assigned. Configure concentrations below or remove assignments to modify.")

    assignments_to_remove = []
    for assignment_key, assignment_data in st.session_state.site_assignments.items():
        sites_list = list(assignment_key)
        elements = assignment_data['elements']

        with st.expander(f"Sites {sites_list}: {', '.join(elements)}", expanded=True):

            if len(elements) == 1:
                assignment_data['concentrations'] = {elements[0]: 1.0}
                st.write(f"**{elements[0]}: 1.000** (single element)")
            else:
                st.write("**Set concentrations:**")

                if not assignment_data['concentrations'] or set(assignment_data['concentrations'].keys()) != set(
                        elements):
                    equal_conc = 1.0 / len(elements)
                    assignment_data['concentrations'] = {elem: equal_conc for elem in elements}

                remaining = 1.0
                new_concentrations = {}

                for i, elem in enumerate(elements[:-1]):
                    current_val = assignment_data['concentrations'].get(elem, 0.0)
                    conc_val = st.slider(
                        f"{elem}:",
                        min_value=0.0,
                        max_value=remaining,
                        value=min(current_val, remaining),
                        step=0.01,
                        format="%.3f",
                        key=f"conc_{assignment_key}_{elem}"
                    )
                    new_concentrations[elem] = conc_val
                    remaining -= conc_val

                last_elem = elements[-1]
                new_concentrations[last_elem] = max(0.0, remaining)
                st.write(f"**{last_elem}: {remaining:.3f}**")

                assignment_data['concentrations'] = new_concentrations


            if st.button(f"❌ Remove Assignment", key=f"remove_{assignment_key}"):
                assignments_to_remove.append(assignment_key)

    for key in assignments_to_remove:
        del st.session_state.site_assignments[key]
        st.rerun()
    if st.session_state.site_assignments:
        st.write("**Current Assignments Summary:**")
        summary_data = []
        for assignment_key, assignment_data in st.session_state.site_assignments.items():
            sites_str = ", ".join(map(str, assignment_key))
            elements_str = ", ".join(assignment_data['elements'])
            conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in assignment_data['concentrations'].items()])

            summary_data.append({
                "Sites": sites_str,
                "Elements": elements_str,
                "Concentrations": conc_str
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    final_assigned_sites = set()
    for assignment_key in st.session_state.site_assignments.keys():
        final_assigned_sites.update(assignment_key)

    final_unassigned_sites = [site['site_index'] for site in all_sites if
                              site['site_index'] not in final_assigned_sites]

    if final_unassigned_sites:
        st.info(f"**Unassigned sites** (keeping original elements): {final_unassigned_sites}")

    chemical_symbols = []
    sublattice_compositions = {}
    first_occurrence_order = []

    for site in all_sites:
        site_idx = site['site_index']

        site_elements = None
        site_concentrations = None

        for assignment_key, assignment_data in st.session_state.site_assignments.items():
            if site_idx in assignment_key:
                site_elements = assignment_data['elements']
                site_concentrations = assignment_data['concentrations']
                break

        if site_elements is None:
            original_element = site['element']
            chemical_symbols.append([original_element])
        else:
            # Assigned site - use specified elements
            # valid_elements = [elem for elem in site_elements if elem != 'X']
            # if not valid_elements:
            #    valid_elements = ['H']  # Dummy for vacancy-only
            # valid_elements = sorted(valid_elements)  # SORT ALPHABETICALLY TO MATCH ICET
            valid_elements = sorted(site_elements)
            chemical_symbols.append(valid_elements)

            if len(site_elements) > 1:
                sorted_site_elements = sorted(site_elements)
                elements_signature = frozenset(sorted_site_elements)

                if elements_signature not in sublattice_compositions:
                    sublattice_compositions[elements_signature] = site_concentrations.copy()
                    first_occurrence_order.append(elements_signature)

    target_concentrations = {}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    sorted_sublattices = []
    for elements_signature in first_occurrence_order:
        elements_list = sorted(list(elements_signature))  # Get sorted elements
        first_element = elements_list[0]  # First element alphabetically
        sorted_sublattices.append((first_element, elements_signature))

    sorted_sublattices.sort(key=lambda x: x[0])

    for i, (first_element, elements_signature) in enumerate(sorted_sublattices):
        if i < len(sublattice_letters):
            sublattice_id = sublattice_letters[i]
            target_concentrations[sublattice_id] = sublattice_compositions[elements_signature]

    is_configured = True

    for assignment_key, assignment_data in st.session_state.site_assignments.items():
        total_conc = sum(assignment_data['concentrations'].values())
        if abs(total_conc - 1.0) > 0.001:
            is_configured = False
            sites_str = ", ".join(map(str, assignment_key))
            st.warning(f"⚠️ Concentrations for sites {sites_str} must sum to 1.0 (currently {total_conc:.3f})")

    if is_configured and target_concentrations:
        st.success("✅ Site assignment configuration is complete!")

        with st.expander("🎯 Generated Configuration", expanded=False):
            st.write("**Chemical Symbols (for ICET ClusterSpace):**")
            st.code(f"chemical_symbols = {chemical_symbols}")

            st.write("**Target Concentrations (for SQS generation):**")
            st.code(f"target_concentrations = {target_concentrations}")

            if target_concentrations:
                st.write("**Sublattice Summary (ordered by first element alphabetically to match ICET):**")
                for sublattice_id, conc in target_concentrations.items():
                    elements = sorted(list(conc.keys()))  # Show sorted elements
                    first_element = elements[0]
                    st.write(f"- **Sublattice {sublattice_id}**: {elements} (first element: '{first_element}')")

                st.info(
                    "💡 **Note**: Sublattices are assigned A, B, C... based on alphabetical order of their first element to match ICET's behavior.")

    elif not st.session_state.site_assignments:
        is_configured = False
        st.info("⚙️ Please assign elements to at least one atomic site.")

    return chemical_symbols, target_concentrations, is_configured


import streamlit as st
import pandas as pd
import numpy as np
from ase.build import make_supercell
import logging
import threading
import queue
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ase import Atoms


def pymatgen_to_ase(structure):
    symbols = [site.specie.symbol if site.is_ordered else site.species.elements[0].symbol for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, True])


def calculate_supercell_factor(transformation_matrix):

    is_diagonal = True
    for i in range(3):
        for j in range(3):
            if i != j and abs(transformation_matrix[i][j]) > 1e-10:
                is_diagonal = False
                break
        if not is_diagonal:
            break

    if is_diagonal:
        return int(round(transformation_matrix[0][0] * transformation_matrix[1][1] * transformation_matrix[2][2]))
    else:
        return int(round(abs(np.linalg.det(transformation_matrix))))


def calculate_achievable_concentrations_sublattice(target_concentrations, chemical_symbols, transformation_matrix,
                                                   primitive_structure):
    supercell_factor = calculate_supercell_factor(transformation_matrix)
    achievable_concentrations = {}
    adjustment_info = []

    sublattice_mapping = {}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    unique_combinations = {}  # {frozenset(elements): [site_indices]}

    for site_idx, site_elements in enumerate(chemical_symbols):
        if len(site_elements) > 1:
            sorted_elements = sorted(site_elements)
            elements_signature = frozenset(sorted_elements)

            if elements_signature not in unique_combinations:
                unique_combinations[elements_signature] = []
            unique_combinations[elements_signature].append(site_idx)

    sorted_combinations = []
    for elements_signature, site_indices in unique_combinations.items():
        elements_list = sorted(list(elements_signature))
        first_element = elements_list[0]
        sorted_combinations.append((first_element, elements_signature, site_indices))

    sorted_combinations.sort(key=lambda x: x[0])

    for i, (first_element, elements_signature, site_indices) in enumerate(sorted_combinations):
        if i < len(sublattice_letters):
            sublattice_letter = sublattice_letters[i]
            sublattice_mapping[sublattice_letter] = {
                'elements': set(elements_signature),
                'site_indices': site_indices
            }

    for sublattice_letter, target_conc in target_concentrations.items():
        if sublattice_letter not in sublattice_mapping:
            st.warning(f"Sublattice {sublattice_letter} not found in chemical symbols")
            achievable_concentrations[sublattice_letter] = {}
            continue

        mapping_info = sublattice_mapping[sublattice_letter]
        sites_in_primitive = len(mapping_info['site_indices'])
        total_sites_in_supercell = sites_in_primitive * supercell_factor

        if total_sites_in_supercell == 0:
            st.warning(f"No sites found for sublattice {sublattice_letter}")
            achievable_concentrations[sublattice_letter] = {}
            continue

        sublattice_achievable = {}
        sublattice_counts = {}

        elements = list(target_conc.keys())
        exact_counts = {}

        for element in elements:
            exact_counts[element] = target_conc[element] * total_sites_in_supercell

        integer_counts = {}
        remainders = {}
        total_assigned = 0

        for element in elements:
            integer_counts[element] = int(exact_counts[element])
            remainders[element] = exact_counts[element] - integer_counts[element]
            total_assigned += integer_counts[element]

        remaining_atoms = total_sites_in_supercell - total_assigned
        if remaining_atoms > 0:
            sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)

            for i in range(remaining_atoms):
                element = sorted_by_remainder[i % len(sorted_by_remainder)][0]
                integer_counts[element] += 1
        elif remaining_atoms < 0:
            sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1])

            for i in range(abs(remaining_atoms)):
                element = sorted_by_remainder[i % len(sorted_by_remainder)][0]
                if integer_counts[element] > 0:
                    integer_counts[element] -= 1

        sublattice_counts = integer_counts

        total_check = sum(sublattice_counts.values())
        if total_check != total_sites_in_supercell:
            st.error(
                f"Atom count mismatch in sublattice {sublattice_letter}: {total_check} != {total_sites_in_supercell}")

        for element, count in sublattice_counts.items():
            sublattice_achievable[element] = count / total_sites_in_supercell

        achievable_concentrations[sublattice_letter] = sublattice_achievable

        for element in target_conc:
            target_val = target_conc[element]
            achievable_val = sublattice_achievable.get(element, 0.0)
            if abs(target_val - achievable_val) > 0.001:
                adjustment_info.append({
                    "Sublattice": sublattice_letter,
                    "Element": element,
                    "Target": f"{target_val:.3f}",
                    "Achievable": f"{achievable_val:.3f}",
                    "Atom Count": sublattice_counts.get(element, 0),
                    "Total Sites": total_sites_in_supercell
                })

    return achievable_concentrations, adjustment_info


def calculate_global_concentrations_from_sublattices(target_concentrations, chemical_symbols, transformation_matrix,
                                                     primitive_structure):
    atoms = pymatgen_to_ase(primitive_structure)
    supercell_factor = calculate_supercell_factor(transformation_matrix)

    global_element_counts = {}

    achievable_concentrations, _ = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    for site_idx, site_elements in enumerate(chemical_symbols):
        if len(site_elements) == 1:
            element = site_elements[0]
            sites_in_supercell = supercell_factor
            if element in global_element_counts:
                global_element_counts[element] += sites_in_supercell
            else:
                global_element_counts[element] = sites_in_supercell
        else:
            sorted_elements = sorted(site_elements)
            elements_signature = frozenset(sorted_elements)

            found_sublattice = None
            for sublattice_letter, achievable_conc in achievable_concentrations.items():
                if set(achievable_conc.keys()) == set(sorted_elements):
                    found_sublattice = sublattice_letter
                    break

            if found_sublattice:
                sites_in_supercell = supercell_factor
                for element, concentration in achievable_concentrations[found_sublattice].items():
                    element_count = concentration * sites_in_supercell
                    if element in global_element_counts:
                        global_element_counts[element] += element_count
                    else:
                        global_element_counts[element] = element_count

    total_sites = len(atoms) * supercell_factor
    global_concentrations = {}

    for element, count in global_element_counts.items():
        global_concentrations[element] = count / total_sites

    return global_concentrations


def display_sublattice_preview(target_concentrations, chemical_symbols, transformation_matrix, primitive_structure):
    try:
        if not target_concentrations:
            st.info("No sublattice concentrations configured yet.")
            return

        achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
            target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
        )

        st.write("**Sublattice Concentrations:**")

        sublattice_data = []
        supercell_factor = int(transformation_matrix[0][0]) * int(transformation_matrix[1][1]) * int(
            transformation_matrix[2][2])

        sublattice_mapping = {}
        sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        unique_combinations = {}  # {frozenset(elements): [site_indices]}

        for site_idx, site_elements in enumerate(chemical_symbols):
            if len(site_elements) > 1:  # Multi-element site
                sorted_elements = sorted(site_elements)
                elements_signature = frozenset(sorted_elements)

                if elements_signature not in unique_combinations:
                    unique_combinations[elements_signature] = []
                unique_combinations[elements_signature].append(site_idx)
        sorted_combinations = []
        for elements_signature, site_indices in unique_combinations.items():
            elements_list = sorted(list(elements_signature))
            first_element = elements_list[0]
            sorted_combinations.append((first_element, elements_signature, site_indices))

        sorted_combinations.sort(key=lambda x: x[0])

        for i, (first_element, elements_signature, site_indices) in enumerate(sorted_combinations):
            if i < len(sublattice_letters):
                sublattice_letter = sublattice_letters[i]
                sublattice_mapping[sublattice_letter] = {
                    'elements': set(elements_signature),
                    'site_indices': site_indices
                }

        for sublattice_letter, target_conc in target_concentrations.items():
            achievable_conc = achievable_concentrations.get(sublattice_letter, {})
            if sublattice_letter in sublattice_mapping:
                sites_count = len(sublattice_mapping[sublattice_letter]['site_indices'])
                total_sublattice_sites = sites_count * supercell_factor

                for element in target_conc.keys():
                    target_frac = target_conc[element]
                    achievable_frac = achievable_conc.get(element, 0.0)
                    atom_count = int(round(achievable_frac * total_sublattice_sites))

                    status = "✅" if abs(target_frac - achievable_frac) < 0.01 else "⚠️"

                    sublattice_data.append({
                        "Sublattice": sublattice_letter,
                        "Element": element,
                        "Target": f"{target_frac:.3f}",
                        "Achievable": f"{achievable_frac:.3f}",
                        "Atoms": atom_count,
                        "Sites": total_sublattice_sites,
                        "Status": status
                    })

        if sublattice_data:
            sublattice_df = pd.DataFrame(sublattice_data)
            st.dataframe(sublattice_df, use_container_width=True)

        global_concentrations = calculate_global_concentrations_from_sublattices(
            target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
        )

        if global_concentrations:
            st.write("**Overall Composition from Sublattices:**")

            atoms = pymatgen_to_ase(primitive_structure)
            total_sites = len(atoms) * supercell_factor

            overall_comp_data = []
            total_global_atoms = 0

            for element in sorted(global_concentrations.keys()):
                global_fraction = global_concentrations[element]
                atom_count = int(round(global_fraction * total_sites))
                total_global_atoms += atom_count

                overall_comp_data.append({
                    "Element": element,
                    "Fraction": f"{global_fraction:.3f}",
                    "Percentage": f"{global_fraction * 100:.1f}%",
                    "Atom Count": atom_count
                })

            if overall_comp_data:
                overall_comp_df = pd.DataFrame(overall_comp_data)
                st.dataframe(overall_comp_df, use_container_width=True)

                st.info(f"**Total atoms in supercell:** {total_global_atoms} / {total_sites}")
        else:
            st.warning("Could not calculate overall composition.")

        # Show adjustment information if any
        #if adjustment_info:
        #    st.warning("⚠️ **Concentration Adjustments Required:**")
        #    adj_df = pd.DataFrame(adjustment_info)
        #    st.dataframe(adj_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating composition preview: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")


class ProgressTracker:

    def __init__(self):
        self.data = {
            'steps': [],
            'scores': [],
            'temperatures': [],
            'accepted_trials': [],
            'timestamps': []
        }
        self.lock = threading.Lock()
        self.last_update = 0

    def add_data_point(self, step, score, temperature, accepted_trials):

        with self.lock:
            self.data['steps'].append(step)
            self.data['scores'].append(score)
            self.data['temperatures'].append(temperature)
            self.data['accepted_trials'].append(accepted_trials)
            self.data['timestamps'].append(time.time())

    def get_data_copy(self):

        with self.lock:
            return {key: val.copy() for key, val in self.data.items()}

    def has_new_data(self, min_interval=0.5):

        current_time = time.time()
        if current_time - self.last_update >= min_interval:
            self.last_update = current_time
            return True
        return False


def create_optimized_chart(progress_data, title="SQS Optimization Progress (Live)"):
    if not progress_data['steps'] or len(progress_data['steps']) < 2:
        return None

    max_points = 1000
    steps = progress_data['steps']
    scores = progress_data['scores']
    temps = progress_data['temperatures']

    if len(steps) > max_points:
        step_size = len(steps) // max_points
        indices = range(0, len(steps), step_size)

        steps = [steps[i] for i in indices]
        scores = [scores[i] for i in indices]
        temps = [temps[i] for i in indices]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=scores,
            mode='lines',
            name='Best Score',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temps,
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2),
            hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title='MC Step',
        height=300,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98),
        # Optimize for performance
        uirevision='constant',  # Maintains zoom/pan state
    )

    fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

    return fig


def thread_for_graph(last_update_time, message_queue, progress_data, progress_placeholder, status_placeholder,
                     chart_placeholder, update_interval):
    if isinstance(last_update_time, list):
        current_last_update = last_update_time[0]
    else:
        current_last_update = last_update_time

    try:
        messages_processed = 0
        max_messages = 20
        chart_updated = False

        while not message_queue.empty() and messages_processed < max_messages:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)

            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])

                if progress_placeholder:
                    progress = min(parsed['current_step'] / max(1, parsed['total_steps'] - 1), 1.0)
                    progress_placeholder.progress(progress)

                if status_placeholder:
                    status_placeholder.text(
                        f"🔄 Step {parsed['current_step']}/{parsed['total_steps']} | "
                        f"Best Score: {parsed['best_score']:.4f} | "
                        f"Temperature: {parsed['temperature']:.3f} | "
                        f"Accepted: {parsed['accepted_trials']}"
                    )

                chart_updated = True

            messages_processed += 1

    except queue.Empty:
        pass

    current_time = time.time()
    if chart_placeholder and chart_updated and (current_time - current_last_update) > update_interval:
        if len(progress_data['steps']) > 1:
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                steps = progress_data['steps']
                scores = progress_data['scores']
                temps = progress_data['temperatures']

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines',
                        name='Best Score',
                        line=dict(color='blue', width=2),
                        hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=temps,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='red', width=2),
                        hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                    ),
                    secondary_y=True
                )

                fig.update_layout(
                    title=dict(
                        text='SQS Optimization Progress (Live)',
                        font=dict(size=16)
                    ),
                    xaxis_title='MC Step',
                    height=300,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98)
                )

                fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
                fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

                chart_placeholder.plotly_chart(fig, use_container_width=True,
                                               key=f"live_chart_{int(current_time * 1000)}")

                if isinstance(last_update_time, list):
                    last_update_time[0] = current_time

            except Exception as e:
                if status_placeholder:
                    status_placeholder.text(f"Continuing optimization... (chart update paused)")


def create_final_chart(progress_data, title="SQS Optimization Results"):
    if not progress_data['steps'] or len(progress_data['steps']) < 2:
        return None

    steps = progress_data['steps']
    scores = progress_data['scores']
    temps = progress_data['temperatures']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=scores,
            mode='lines',
            name='Best Score',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temps,
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2),
            hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title='MC Step',
        height=300,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )

    fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

    return fig


def generate_sqs_with_icet_progress_multi(primitive_structure, target_concentrations, transformation_matrix,
                                          cutoffs, method="monte_carlo", n_steps=10000, random_seed=42,
                                          progress_placeholder=None, chart_placeholder=None, status_placeholder=None):
    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, achievable_counts = calculate_achievable_concentrations(
        target_concentrations, total_sites)

    concentration_adjusted = False
    for element in target_concentrations:
        if abs(target_concentrations[element] - achievable_concentrations[element]) > 0.001:
            concentration_adjusted = True
            break

    #if concentration_adjusted:
    #    st.warning("⚠️ **Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_data = []
    #    for element in sorted(target_concentrations.keys()):
    #        adj_data.append({
    #            "Element": element,
    #            "Target": f"{target_concentrations[element]:.3f}",
    #            "Achievable": f"{achievable_concentrations[element]:.3f}",
    #            "Atom Count": achievable_counts[element]
    #        })
    #    adj_df = pd.DataFrame(adj_data)
    #    st.dataframe(adj_df, use_container_width=True)

    all_elements = list(achievable_concentrations.keys())
    chemical_symbols = [all_elements for _ in range(len(atoms))]

    cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            else:
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph_multi_run(last_update_time, message_queue, progress_data, progress_placeholder,
                                   status_placeholder, chart_placeholder, update_interval)
        time.sleep(0.1)

    thread.join()
    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Run completed! Final step: {final_step}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        try:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            steps = progress_data['steps']
            scores = progress_data['scores']
            temps = progress_data['temperatures']

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=scores,
                    mode='lines',
                    name='Best Score',
                    line=dict(color='blue', width=1),
                    hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=temps,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=1),
                    hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                ),
                secondary_y=True
            )
            font_size = 14
            fig.update_layout(
                title=dict(
                    text='✅ Final SQS Optimization Results',
                    font=dict(size=font_size, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                hovermode='x unified',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    font=dict(size=font_size, family="Arial Black")
                ),
                font=dict(size=font_size, family="Arial"),
                xaxis=dict(
                    title_font=dict(size=font_size, family="Arial Black"),
                    tickfont=dict(size=font_size, family="Arial")
                ),
                yaxis=dict(
                    title_font=dict(size=font_size, family="Arial Black"),
                    tickfont=dict(size=font_size, family="Arial")
                )
            )

            fig.update_yaxes(
                title_text="Best Score",
                secondary_y=False,
                color='blue',
                title_font=dict(size=font_size, family="Arial Black"),
                tickfont=dict(size=font_size, family="Arial")
            )
            fig.update_yaxes(
                title_text="Temperature",
                secondary_y=True,
                color='red',
                title_font=dict(size=font_size, family="Arial Black"),
                tickfont=dict(size=font_size, family="Arial")
            )

            final_chart_key = f"final_multi_chart_{getattr(st.session_state, 'current_multi_run', 0)}_{int(time.time() * 1000)}"
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=final_chart_key)

        except Exception as e:
            st.warning(f"Could not update final chart: {e}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data


from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.structure import PartialRadialDistributionFunction
from itertools import combinations
from collections import defaultdict
import plotly.graph_objects as go


def calculate_sqs_prdf(structure, cutoff=10.0, bin_size=0.1):
    try:

        elements = list(set([site.specie.symbol for site in structure if site.is_ordered]))

        species_combinations = list(combinations(elements, 2)) + [(s, s) for s in elements]

        prdf_featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
        prdf_featurizer.fit([structure])

        prdf_data = prdf_featurizer.featurize(structure)
        feature_labels = prdf_featurizer.feature_labels()

        prdf_dict = defaultdict(list)
        distance_dict = {}

        for i, label in enumerate(feature_labels):
            parts = label.split(" PRDF r=")
            element_pair = tuple(parts[0].split("-"))
            distance_range = parts[1].split("-")
            bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2
            prdf_dict[element_pair].append(prdf_data[i])

            if element_pair not in distance_dict:
                distance_dict[element_pair] = []
            distance_dict[element_pair].append(bin_center)

        return prdf_dict, distance_dict, species_combinations

    except Exception as e:
        st.error(f"Error calculating PRDF: {e}")
        return None, None, None


def calculate_and_display_sqs_prdf(sqs_structure, cutoff=10.0, bin_size=0.1):
    try:
        with st.expander("📊 PRDF Analysis of Generated SQS", expanded = True):
            with st.spinner("Calculating PRDF..."):
                prdf_dict, distance_dict, species_combinations = calculate_sqs_prdf(
                    sqs_structure, cutoff=cutoff, bin_size=bin_size
                )

                if prdf_dict is not None:
                    import matplotlib.pyplot as plt
                    import numpy as np

                    colors = plt.cm.tab10.colors

                    def rgb_to_hex(color):
                        return '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                    font_dict = dict(size=18, color="black")

                    fig_combined = go.Figure()

                    for idx, (pair, prdf_values) in enumerate(prdf_dict.items()):
                        hex_color = rgb_to_hex(colors[idx % len(colors)])

                        fig_combined.add_trace(go.Scatter(
                            x=distance_dict[pair],
                            y=prdf_values,
                            mode='lines+markers',
                            name=f"{pair[0]}-{pair[1]}",
                            line=dict(color=hex_color, width=2),
                            marker=dict(size=6)
                        ))

                    fig_combined.update_layout(
                        title={'text': "SQS PRDF: All Element Pairs", 'font': font_dict},
                        xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                        yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                        hovermode='x',
                        font=font_dict,
                        xaxis=dict(tickfont=font_dict),
                        yaxis=dict(tickfont=font_dict, range=[0, None]),
                        hoverlabel=dict(font=font_dict),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=16)
                        )
                    )

                    st.plotly_chart(fig_combined, use_container_width=True)

                    import pandas as pd
                    import base64

                    for pair, prdf_values in prdf_dict.items():
                        df = pd.DataFrame()
                        df["Distance (Å)"] = distance_dict[pair]
                        df["PRDF"] = prdf_values

                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        filename = f"SQS_{pair[0]}_{pair[1]}_prdf.csv"
                        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {pair[0]}-{pair[1]} PRDF data</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    return True

                else:
                    st.error("Failed to calculate PRDF")
                    return False

    except Exception as e:
        st.error(f"Error calculating PRDF: {e}")
        return False


from io import StringIO
from ase.constraints import FixAtoms


def create_sqs_download_section(result, selected_file):
    st.subheader("📥 Download SQS Structure")

    col_download_format, col_download_button = st.columns([1, 1])

    with col_download_format:
        file_format = st.radio(
            f"Select file **format**",
            ("CIF", "VASP", "LAMMPS", "XYZ"),
            horizontal=True,
            key="sqs_download_format"
        )

    file_content = None
    download_file_name = None
    mime = "text/plain"

    sqs_structure = result['structure']

    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter

            download_file_name = f"SQS_{selected_file.split('.')[0]}.cif"
            mime = "chemical/x-cif"
            file_content = result['cif_content']

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.poscar"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            col_vasp1, col_vasp2 = st.columns([1, 1])
            with col_vasp1:
                use_fractional = st.checkbox(
                    "Output POSCAR with fractional coordinates",
                    value=True,
                    key="sqs_poscar_fractional"
                )

            with col_vasp2:
                from ase.constraints import FixAtoms
                use_selective_dynamics = st.checkbox(
                    "Include Selective dynamics (all atoms free)",
                    value=False,
                    key="sqs_poscar_sd"
                )
                if use_selective_dynamics:
                    constraint = FixAtoms(indices=[])
                    current_ase_structure.set_constraint(constraint)

            out = StringIO()
            write(out, current_ase_structure, format="vasp", direct=use_fractional, sort=True)
            file_content = out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.lmp"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            st.markdown("**LAMMPS Export Options**")
            col_lmp1, col_lmp2 = st.columns([1, 1])

            with col_lmp1:
                atom_style = st.selectbox(
                    "Select atom_style",
                    ["atomic", "charge", "full"],
                    index=0,
                    key="sqs_lammps_atom_style"
                )
                units = st.selectbox(
                    "Select units",
                    ["metal", "real", "si"],
                    index=0,
                    key="sqs_lammps_units"
                )

            with col_lmp2:
                include_masses = st.checkbox(
                    "Include atomic masses",
                    value=True,
                    key="sqs_lammps_masses"
                )
                force_skew = st.checkbox(
                    "Force triclinic cell (skew)",
                    value=False,
                    key="sqs_lammps_skew"
                )

            out = StringIO()
            write(
                out,
                current_ase_structure,
                format="lammps-data",
                atom_style=atom_style,
                units=units,
                masses=include_masses,
                force_skew=force_skew
            )
            file_content = out.getvalue()

        elif file_format == "XYZ":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.xyz"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            out = StringIO()
            write(out, current_ase_structure, format="xyz")
            file_content = out.getvalue()

    except Exception as e:
        st.error(f"Error generating {file_format} file: {e}")
        st.error(
            f"There was an error processing the SQS structure for {file_format} format. "
            f"Please try a different format or check the structure validity."
        )

    with col_download_button:
        if file_content is not None:
            st.download_button(
                label=f"📥 Download {file_format} file",
                data=file_content,
                file_name=download_file_name,
                type="primary",
                mime=mime,
                key=f"sqs_download_{file_format.lower()}"
            )
        else:
            st.info(f"Select {file_format} format to enable download")


def pymatgen_to_ase(structure):
    from ase import Atoms
    import numpy as np

    symbols = [str(site.specie) for site in structure]
    positions = [site.coords for site in structure]
    cell = structure.lattice.matrix

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )
    return atoms



def thread_for_graph_multi_run(last_update_time, message_queue, progress_data, progress_placeholder, status_placeholder,
                               chart_placeholder, update_interval):
    if isinstance(last_update_time, list):
        current_last_update = last_update_time[0]
    else:
        current_last_update = last_update_time

    try:
        messages_processed = 0
        max_messages = 20
        chart_updated = False
        latest_parsed = None

        while not message_queue.empty() and messages_processed < max_messages:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)

            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
                latest_parsed = parsed

                if progress_placeholder:
                    try:
                        progress = min(parsed['current_step'] / max(1, parsed['total_steps'] - 1), 1.0)
                        progress_placeholder.progress(progress)
                    except:
                        pass

                if status_placeholder:
                    try:
                        status_placeholder.text(
                            f"🔄 MC Step {parsed['current_step']}/{parsed['total_steps']} | "
                            f"Best Score: {parsed['best_score']:.4f} | "
                            f"Temperature: {parsed['temperature']:.3f} | "
                            f"Accepted: {parsed['accepted_trials']}"
                        )
                    except Exception as e:
                        try:
                            status_placeholder.text(f"🔄 MC Step {parsed['current_step']}/{parsed['total_steps']}")
                        except:
                            pass

                chart_updated = True

            messages_processed += 1

    except queue.Empty:
        pass
    except Exception as e:
        if status_placeholder and latest_parsed:
            try:
                status_placeholder.text(f"🔄 MC Step {latest_parsed['current_step']}/{latest_parsed['total_steps']}")
            except:
                pass

    current_time = time.time()
    if chart_placeholder and chart_updated and (current_time - current_last_update) > update_interval:
        if len(progress_data['steps']) > 1:
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                steps = progress_data['steps']
                scores = progress_data['scores']
                temps = progress_data['temperatures']

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines',
                        name='Best Score',
                        line=dict(color='blue', width=4),
                        hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=temps,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='red', width=4),
                        hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                    ),
                    secondary_y=True
                )

                font_ss = 14
                fig.update_layout(
                    title=dict(
                        text='SQS Optimization Progress (Live)',
                        font=dict(size=font_ss, family="Arial Black")
                    ),
                    xaxis_title='MC Step',
                    height=350,
                    hovermode='x unified',
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        font=dict(size=font_ss, family="Arial Black")
                    ),
                    font=dict(size=font_ss, family="Arial"),
                    xaxis=dict(
                        title_font=dict(size=font_ss, family="Arial Black"),
                        tickfont=dict(size=font_ss, family="Arial")
                    ),
                    yaxis=dict(
                        title_font=dict(size=font_ss, family="Arial Black"),


                        tickfont=dict(size=font_ss, family="Arial")
                    )
                )

                fig.update_yaxes(
                    title_text="Best Score",
                    secondary_y=False,
                    color='blue',
                    title_font=dict(size=font_ss, family="Arial Black"),
                    tickfont=dict(size=font_ss, family="Arial")
                )
                fig.update_yaxes(
                    title_text="Temperature",
                    secondary_y=True,
                    color='red',
                    title_font=dict(size=font_ss, family="Arial Black"),
                    tickfont=dict(size=font_ss, family="Arial")
                )

                chart_key = f"multi_run_chart_{getattr(st.session_state, 'current_multi_run', 0)}_{int(current_time)}"
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=chart_key)

                if isinstance(last_update_time, list):
                    last_update_time[0] = current_time

            except Exception as e:
                if status_placeholder and latest_parsed:
                    try:
                        status_placeholder.text(f"🔄 MC Step {latest_parsed['current_step']}/{latest_parsed['total_steps']} | Score: {latest_parsed['best_score']:.4f}")
                    except:
                        pass

    return latest_parsed


def generate_structure_file_content_multi(structure, file_format):
    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter
            cif_writer = CifWriter(structure)
            return cif_writer.__str__()

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="vasp", direct=True, sort=True)
            return out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="lammps-data", atom_style="atomic", units="metal")
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
            return "Unsupported format"

    except Exception as e:
        return f"Error generating {file_format}: {str(e)}"




def generate_sqs_with_icet_progress_sublattice(primitive_structure, chemical_symbols, target_concentrations,
                                               transformation_matrix, cutoffs, method="monte_carlo",
                                               n_steps=10000, random_seed=42, progress_placeholder=None,
                                               chart_placeholder=None, status_placeholder=None):
    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    #if adjustment_info:
    #    st.warning(
    #        "⚠️ **Sublattice Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_df = pd.DataFrame(adjustment_info)
    #    st.dataframe(adj_df, use_container_width=True)

    try:
        cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)
    except Exception as e:
        st.error(f"Error creating ClusterSpace: {e}")
        st.write(f"Chemical symbols: {chemical_symbols}")
        st.write(f"Atoms symbols: {atoms.get_chemical_symbols()}")
        raise

    # Display sublattice information
    st.write("**Sublattice Configuration:**")
    sublattice_info_data = []
    for sublattice_id, sublattice_conc in achievable_concentrations.items():
        elements = list(sublattice_conc.keys())
        conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in sublattice_conc.items()])
        sublattice_info_data.append({
            "Sublattice": sublattice_id,
            "Elements": ", ".join(elements),
            "Concentrations": conc_str
        })

    if sublattice_info_data:
        sublattice_df = pd.DataFrame(sublattice_info_data)
        st.dataframe(sublattice_df, use_container_width=True)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        time.sleep(5)
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            elif method == "enumeration":
                return generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations
                )
            else:
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph(
            last_update_time,
            message_queue,
            progress_data,
            progress_placeholder,
            status_placeholder,
            chart_placeholder,
            update_interval
        )
        time.sleep(0.1)

    thread.join()

    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Generation completed! Final step: {final_step+1000}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        with chart_placeholder.container():
            st.write("**Final SQS Optimization Results:**")
            final_fig = create_final_chart(progress_data, title="Final SQS Optimization Results")
            if final_fig:
                current_run = getattr(st.session_state, 'current_multi_run', 0)
                chart_key = f"sqs_sublattice_final_chart_run_{current_run}_{int(time.time() * 1000)}"
                st.plotly_chart(final_fig, use_container_width=True, key=chart_key)
            else:
                st.info("Optimization completed - see live chart above for progress details.")

            if progress_data['scores']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Steps", 1000*len(progress_data['steps']))
                with col2:
                    st.metric("Best Score", f"{min(progress_data['scores']):.4f}")
                with col3:
                    st.metric("Final Score", f"{progress_data['scores'][-1]:.4f}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    if log_handler in icet_logger.handlers:
        icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data

def generate_sqs_with_icet_progress_sublattice_multi(primitive_structure, chemical_symbols, target_concentrations,
                                                     transformation_matrix, cutoffs, method="monte_carlo",
                                                     n_steps=10000, random_seed=42, progress_placeholder=None,
                                                     chart_placeholder=None, status_placeholder=None):

    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    #if adjustment_info:
    #    st.warning(
    #        "⚠️ **Sublattice Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_df = pd.DataFrame(adjustment_info)
    #    st.dataframe(adj_df, use_container_width=True)

    try:
        cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)
    except Exception as e:
        st.error(f"Error creating ClusterSpace: {e}")
        st.write(f"Chemical symbols: {chemical_symbols}")
        st.write(f"Atoms symbols: {atoms.get_chemical_symbols()}")
        raise

    # Display sublattice information
    #st.write("**Sublattice Configuration:**")
    #sublattice_info_data = []
    #for sublattice_id, sublattice_conc in achievable_concentrations.items():
    #    elements = list(sublattice_conc.keys())
    #    conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in sublattice_conc.items()])
    #    sublattice_info_data.append({
    #        "Sublattice": sublattice_id,
    #        "Elements": ", ".join(elements),
    #        "Concentrations": conc_str
    #    })

    #if sublattice_info_data:
    #    sublattice_df = pd.DataFrame(sublattice_info_data)
    #    st.dataframe(sublattice_df, use_container_width=True)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        time.sleep(5)
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            elif method == "enumeration":
                return generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations
                )
            else:  # monte_carlo
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph_multi_run(
            last_update_time,
            message_queue,
            progress_data,
            progress_placeholder,
            status_placeholder,
            chart_placeholder,
            update_interval
        )
        time.sleep(0.1)

    thread.join()

    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Run completed! Final step: {final_step}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        try:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            steps = progress_data['steps']
            scores = progress_data['scores']
            temps = progress_data['temperatures']

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=scores,
                    mode='lines',
                    name='Best Score',
                    line=dict(color='blue', width=1),
                    hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=temps,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=1),
                    hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                ),
                secondary_y=True
            )
            font_sizz = 12
            fig.update_layout(
                title=dict(
                    text='✅ Final SQS Optimization Results (Sublattice)',
                    font=dict(size=font_sizz, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                hovermode='x unified',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    font=dict(size=font_sizz, family="Arial Black")
                ),
                font=dict(size=font_sizz, family="Arial"),
                xaxis=dict(
                    title_font=dict(size=font_sizz, family="Arial Black"),
                    tickfont=dict(size=font_sizz, family="Arial")
                ),
                yaxis=dict(
                    title_font=dict(size=font_sizz, family="Arial Black"),
                    tickfont=dict(size=font_sizz, family="Arial")
                )
            )

            fig.update_yaxes(
                title_text="Best Score",
                secondary_y=False,
                color='blue',
                title_font=dict(size=font_sizz, family="Arial Black"),
                tickfont=dict(size=font_sizz, family="Arial")
            )
            fig.update_yaxes(
                title_text="Temperature",
                secondary_y=True,
                color='red',
                title_font=dict(size=font_sizz, family="Arial Black"),
            )

            current_run = getattr(st.session_state, 'current_multi_run', 0)
            final_chart_key = f"final_sublattice_multi_chart_run_{current_run}_{int(time.time() * 1000)}"
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=final_chart_key)

        except Exception as e:
            st.warning(f"Could not update final chart: {e}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    if log_handler in icet_logger.handlers:
        icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data


SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1", 3: "P2", 4: "P21", 5: "C2", 6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc", 10: "P2/m",
    11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c", 16: "P222", 17: "P2221", 18: "P21212", 19: "P212121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I212121", 25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21", 30: "Pnc2",
    31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2", 35: "Cmm2", 36: "Cmc21", 37: "Ccc2", 38: "Amm2", 39: "Aem2", 40: "Ama2",
    41: "Aea2", 42: "Fmm2", 43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2", 47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban",
    51: "Pmma", 52: "Pnna", 53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm", 59: "Pmmn", 60: "Pbcn",
    61: "Pbca", 62: "Pnma", 63: "Cmcm", 64: "Cmca", 65: "Cmmm", 66: "Cccm", 67: "Cmma", 68: "Ccca", 69: "Fmmm", 70: "Fddd",
    71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma", 75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4", 80: "I41",
    81: "P-4", 82: "I-4", 83: "P4/m", 84: "P42/m", 85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a", 89: "P422", 90: "P4212",
    91: "P4122", 92: "P41212", 93: "P4222", 94: "P42212", 95: "P4322", 96: "P43212", 97: "I422", 98: "I4122", 99: "P4mm", 100: "P4bm",
    101: "P42cm", 102: "P42nm", 103: "P4cc", 104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "P42mm", 108: "P42cm", 109: "I4mm", 110: "I4cm",
    111: "I41md", 112: "I41cd", 113: "P-42m", 114: "P-42c", 115: "P-421m", 116: "P-421c", 117: "P-4m2", 118: "P-4c2", 119: "P-4b2", 120: "P-4n2",
    121: "I-4m2", 122: "I-4c2", 123: "I-42m", 124: "I-42d", 125: "P4/mmm", 126: "P4/mcc", 127: "P4/nbm", 128: "P4/nnc", 129: "P4/mbm", 130: "P4/mnc",
    131: "P4/nmm", 132: "P4/ncc", 133: "P42/mmc", 134: "P42/mcm", 135: "P42/nbc", 136: "P42/mnm", 137: "P42/mbc", 138: "P42/mnm", 139: "I4/mmm", 140: "I4/mcm",
    141: "I41/amd", 142: "I41/acd", 143: "P3", 144: "P31", 145: "P32", 146: "R3", 147: "P-3", 148: "R-3", 149: "P312", 150: "P321",
    151: "P3112", 152: "P3121", 153: "P3212", 154: "P3221", 155: "R32", 156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c", 160: "R3m",
    161: "R3c", 162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1", 166: "R-3m", 167: "R-3c", 168: "P6", 169: "P61", 170: "P65",
    171: "P62", 172: "P64", 173: "P63", 174: "P-6", 175: "P6/m", 176: "P63/m", 177: "P622", 178: "P6122", 179: "P6522", 180: "P6222",
    181: "P6422", 182: "P6322", 183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc", 187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P63/mcm", 194: "P63/mmc", 195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213", 200: "Pm-3",
    201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3", 205: "Pa-3", 206: "Ia-3", 207: "P432", 208: "P4232", 209: "F432", 210: "F4132",
    211: "I432", 212: "P4332", 213: "P4132", 214: "I4132", 215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c", 220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m", 225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c", 229: "Im-3m", 230: "Ia-3d"
}


def get_formula_type(formula):
    elements = []
    counts = []

    import re
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    for element, count in matches:
        elements.append(element)
        counts.append(int(count) if count else 1)

    if len(elements) == 1:
        return "A"

    elif len(elements) == 2:
        # Binary compounds
        if counts[0] == 1 and counts[1] == 1:
            return "AB"
        elif counts[0] == 1 and counts[1] == 2:
            return "AB2"
        elif counts[0] == 2 and counts[1] == 1:
            return "A2B"
        elif counts[0] == 1 and counts[1] == 3:
            return "AB3"
        elif counts[0] == 3 and counts[1] == 1:
            return "A3B"
        elif counts[0] == 1 and counts[1] == 4:
            return "AB4"
        elif counts[0] == 4 and counts[1] == 1:
            return "A4B"
        elif counts[0] == 1 and counts[1] == 5:
            return "AB5"
        elif counts[0] == 5 and counts[1] == 1:
            return "A5B"
        elif counts[0] == 1 and counts[1] == 6:
            return "AB6"
        elif counts[0] == 6 and counts[1] == 1:
            return "A6B"
        elif counts[0] == 2 and counts[1] == 3:
            return "A2B3"
        elif counts[0] == 3 and counts[1] == 2:
            return "A3B2"
        elif counts[0] == 2 and counts[1] == 5:
            return "A2B5"
        elif counts[0] == 5 and counts[1] == 2:
            return "A5B2"
        elif counts[0] == 1 and counts[1] == 12:
            return "AB12"
        elif counts[0] == 12 and counts[1] == 1:
            return "A12B"
        elif counts[0] == 2 and counts[1] == 17:
            return "A2B17"
        elif counts[0] == 17 and counts[1] == 2:
            return "A17B2"
        elif counts[0] == 3 and counts[1] == 4:
            return "A3B4"
        else:
            return f"A{counts[0]}B{counts[1]}"

    elif len(elements) == 3:
        # Ternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1:
            return "ABC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3:
            return "ABC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1:
            return "AB3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1:
            return "A3BC"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4:
            return "AB2C4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4:
            return "A2BC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 2:
            return "AB4C2"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1:
            return "A2B4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 2:
            return "A4BC2"
        elif counts[0] == 4 and counts[1] == 2 and counts[2] == 1:
            return "A4B2C"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1:
            return "AB2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1:
            return "A2BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2:
            return "ABC2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4:
            return "ABC4"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1:
            return "AB4C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1:
            return "A4BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 5:
            return "ABC5"
        elif counts[0] == 1 and counts[1] == 5 and counts[2] == 1:
            return "AB5C"
        elif counts[0] == 5 and counts[1] == 1 and counts[2] == 1:
            return "A5BC"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 6:
            return "ABC6"
        elif counts[0] == 1 and counts[1] == 6 and counts[2] == 1:
            return "AB6C"
        elif counts[0] == 6 and counts[1] == 1 and counts[2] == 1:
            return "A6BC"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 1:
            return "A2B2C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 2:
            return "A2BC2"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 2:
            return "AB2C2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1:
            return "A3B2C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2:
            return "A3BC2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2:
            return "AB3C2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1:
            return "A2B3C"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3:
            return "A2BC3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3:
            return "AB2C3"
        elif counts[0] == 3 and counts[1] == 3 and counts[2] == 1:
            return "A3B3C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 3:
            return "A3BC3"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 3:
            return "AB3C3"
        elif counts[0] == 4 and counts[1] == 3 and counts[2] == 1:
            return "A4B3C"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 3:
            return "A4BC3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 3:
            return "AB4C3"
        elif counts[0] == 3 and counts[1] == 4 and counts[2] == 1:
            return "A3B4C"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 4:
            return "A3BC4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "AB3C4"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 4:
            return "ABC6"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 7:
            return "A2B2C7"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}"

    elif len(elements) == 4:
        # Quaternary compounds
        if counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "ABCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "ABCD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "ABC3D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "AB3CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A3BCD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "ABCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "ABC4D"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "AB4CD"
        elif counts[0] == 4 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A4BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 4:
            return "AB2CD4"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 4:
            return "A2BCD4"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 4:
            return "ABC2D4"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 4 and counts[3] == 1:
            return "AB2C4D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 4 and counts[3] == 1:
            return "A2BC4D"
        elif counts[0] == 2 and counts[1] == 4 and counts[2] == 1 and counts[3] == 1:
            return "A2B4CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 1:
            return "A2BCD"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "AB2CD"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "ABC2D"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "ABCD2"
        elif counts[0] == 3 and counts[1] == 2 and counts[2] == 1 and counts[3] == 1:
            return "A3B2CD"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 2 and counts[3] == 1:
            return "A3BC2D"
        elif counts[0] == 3 and counts[1] == 1 and counts[2] == 1 and counts[3] == 2:
            return "A3BCD2"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 2 and counts[3] == 1:
            return "AB3C2D"
        elif counts[0] == 1 and counts[1] == 3 and counts[2] == 1 and counts[3] == 2:
            return "AB3CD2"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 2:
            return "ABC3D2"
        elif counts[0] == 2 and counts[1] == 3 and counts[2] == 1 and counts[3] == 1:
            return "A2B3CD"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 3 and counts[3] == 1:
            return "A2BC3D"
        elif counts[0] == 2 and counts[1] == 1 and counts[2] == 1 and counts[3] == 3:
            return "A2BCD3"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 3 and counts[3] == 1:
            return "AB2C3D"
        elif counts[0] == 1 and counts[1] == 2 and counts[2] == 1 and counts[3] == 3:
            return "AB2CD3"
        elif counts[0] == 1 and counts[1] == 1 and counts[2] == 2 and counts[3] == 3:
            return "ABC2D3"
        elif counts[0] == 1 and counts[1] == 4 and counts[2] == 1 and counts[3] == 6:
            return "A1B4C1D6"
        elif counts[0] == 5 and counts[1] == 3 and counts[2] == 1 and counts[3] == 13:
            return "A5B3C1D13"
        elif counts[0] == 2 and counts[1] == 2 and counts[2] == 4 and counts[3] == 9:
            return "A2B2C4D9"

        elif counts == [3, 2, 1, 4]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C1D4"
        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}"

    elif len(elements) == 5:
        # Five-element compounds (complex minerals like apatite)
        if counts == [1, 1, 1, 1, 1]:
            return "ABCDE"
        elif counts == [10, 6, 2, 31, 1]:  # Apatite-like: Ca10(PO4)6(OH)2
            return "A10B6C2D31E"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13DE"
        elif counts == [5, 3, 13, 1, 1]:  # Simplified apatite: Ca5(PO4)3OH
            return "A5B3C13"
        elif counts == [3, 2, 3, 12, 1]:  # Garnet-like: Ca3Al2Si3O12
            return "A3B2C3D12E"

        else:
            return f"A{counts[0]}B{counts[1]}C{counts[2]}D{counts[3]}E{counts[4]}"

    elif len(elements) == 6:
        # Six-element compounds (very complex minerals)
        if counts == [1, 1, 1, 1, 1, 1]:
            return "ABCDEF"
        elif counts == [1, 1, 2, 6, 1, 1]:  # Complex silicate-like
            return "ABC2D6EF"
        else:
            # For 6+ elements, use a more compact notation
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, ...
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)

    else:
        if len(elements) <= 10:
            element_count_pairs = []
            for i, count in enumerate(counts):
                element_letter = chr(65 + i)  # A, B, C, D, E, F, G, H, I, J
                if count == 1:
                    element_count_pairs.append(element_letter)
                else:
                    element_count_pairs.append(f"{element_letter}{count}")
            return "".join(element_count_pairs)
        else:
            return "Complex"
def identify_structure_type(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()

        formula = structure.composition.reduced_formula
        formula_type = get_formula_type(formula)
       # print("------")
        print(formula)
       # print(formula_type)
        #print(spg_number)
        if spg_number in STRUCTURE_TYPES and spg_number == 62 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Aragonite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==167 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "CaCO3":
          #  print("YES")
          # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Calcite (CaCO3)**"
        elif spg_number in STRUCTURE_TYPES and spg_number ==227 and formula_type in STRUCTURE_TYPES[spg_number] and formula == "SiO2":
           # print("YES")
           # print(spg_number)
           # print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**β - Cristobalite (SiO2)**"
        elif formula == "C" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Graphite**"
        elif formula == "MoS2" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**MoS2 Type**"
        elif formula == "NiAs" and spg_number in STRUCTURE_TYPES and spg_number ==194 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**Nickeline (NiAs)**"
        elif formula == "ReO3" and spg_number in STRUCTURE_TYPES and spg_number ==221 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**ReO3 type**"
        elif formula == "TlI" and spg_number in STRUCTURE_TYPES and spg_number ==63 :
            print("YES")
            print(spg_number)
            print(formula_type)
            #structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**TlI structure**"
        elif spg_number in STRUCTURE_TYPES and formula_type in STRUCTURE_TYPES[
            spg_number]:
           # print("YES")
            structure_type = STRUCTURE_TYPES[spg_number][formula_type]
            return f"**{structure_type}**"

        pearson = f"{crystal_system[0]}{structure.num_sites}"
        return f"**{crystal_system.capitalize()}** (Formula: {formula_type}, Pearson: {pearson})"

    except Exception as e:
        return f"Error identifying structure: {str(e)}"
STRUCTURE_TYPES = {
    # Cubic Structures
    225: {  # Fm-3m
        "A": "FCC (Face-centered cubic)",
        "AB": "Rock Salt (NaCl)",
        "AB2": "Fluorite (CaF2)",
        "A2B": "Anti-Fluorite",
        "AB3": "Cu3Au (L1₂)",
        "A3B": "AuCu3 type",
        "ABC": "Half-Heusler (C1b)",
        "AB6": "K2PtCl6 (cubic antifluorite)",
    },
    92: {
        "AB2": "α-Cristobalite (SiO2)"
    },
    229: {  # Im-3m
        "A": "BCC (Body-centered cubic)",
        "AB12": "NaZn13 type",
        "AB": "Tungsten carbide (WC)"
    },
    221: {  # Pm-3m
        "A": "Simple cubic (SC)",
        "AB": "Cesium Chloride (CsCl)",
        "ABC3": "Perovskite (Cubic, ABO3)",
        "AB3": "Cu3Au type",
        "A3B": "Cr3Si (A15)",
        #"AB6": "ReO3 type"
    },
    227: {  # Fd-3m
        "A": "Diamond cubic",

        "AB2": "Fluorite-like",
        "AB2C4": "Normal spinel",
        "A3B4": "Inverse spinel",
        "AB2C4": "Spinel",
        "A8B": "Gamma-brass",
        "AB2": "β - Cristobalite (SiO2)",
        "A2B2C7": "Pyrochlore"
    },
    55: {  # Pbca
        "AB2": "Brookite (TiO₂ polymorph)"
    },
    216: {  # F-43m
        "AB": "Zinc Blende (Sphalerite)",
        "A2B": "Antifluorite"
    },
    215: {  # P-43m
        "ABC3": "Inverse-perovskite",
        "AB4": "Half-anti-fluorite"
    },
    223: {  # Pm-3n
        "AB": "α-Mn structure",
        "A3B": "Cr3Si-type"
    },
    230: {  # Ia-3d
        "A3B2C1D4": "Garnet structure ((Ca,Mg,Fe)3(Al,Fe)2(SiO4)3)",
        "AB2": "Pyrochlore"
    },
    217: {  # I-43m
        "A12B": "α-Mn structure"
    },
    219: {  # F-43c
        "AB": "Sodium thallide"
    },
    205: {  # Pa-3
        "A2B": "Cuprite (Cu2O)",
        "AB6": "ReO3 structure",
        "AB2": "Pyrite (FeS2)",
    },
    156: {
        "AB2": "CdI2 type",
    },
    # Hexagonal Structures
    194: {  # P6_3/mmc
        "AB": "Wurtzite (high-T)",
        "AB2": "AlB2 type (hexagonal)",
        "A3B": "Ni3Sn type",
        "A3B": "DO19 structure (Ni3Sn-type)",
        "A": "Graphite (hexagonal)",
        "A": "HCP (Hexagonal close-packed)",
        #"AB2": "MoS2 type",
    },
    186: {  # P6_3mc
        "AB": "Wurtzite (ZnS)",
    },
    191: {  # P6/mmm


        "AB2": "AlB2 type",
        "AB5": "CaCu5 type",
        "A2B17": "Th2Ni17 type"
    },
    193: {  # P6_3/mcm
        "A3B": "Na3As structure",
        "ABC": "ZrBeSi structure"
    },
   # 187: {  # P-6m2
#
 #   },
    164: {  # P-3m1
        "AB2": "CdI2 type",
        "A": "Graphene layers"
    },
    166: {  # R-3m
        "A": "Rhombohedral",
        "A2B3": "α-Al2O3 type",
        "ABC2": "Delafossite (CuAlO2)"
    },
    160: {  # R3m
        "A2B3": "Binary tetradymite",
        "AB2": "Delafossite"
    },

    # Tetragonal Structures
    139: {  # I4/mmm
        "A": "Body-centered tetragonal",
        "AB": "β-Tin",
        "A2B": "MoSi2 type",
        "A3B": "Ni3Ti structure"
    },
    136: {  # P4_2/mnm
        "AB2": "Rutile (TiO2)"
    },
    123: {  # P4/mmm
        "AB": "γ-CuTi",
        "AB": "CuAu (L10)"
    },
    140: {  # I4/mcm
        "AB2": "Anatase (TiO2)",
        "A": "β-W structure"
    },
    141: {  # I41/amd
        "AB2": "Anatase (TiO₂)",
        "A": "α-Sn structure",
        "ABC4": "Zircon (ZrSiO₄)"
    },
    122: {  # P-4m2
        "ABC2": "Chalcopyrite (CuFeS2)"
    },
    129: {  # P4/nmm
        "AB": "PbO structure"
    },

    # Orthorhombic Structures
    62: {  # Pnma
        "ABC3": "Aragonite (CaCO₃)",
        "AB2": "Cotunnite (PbCl2)",
        "ABC3": "Perovskite (orthorhombic)",
        "A2B": "Fe2P type",
        "ABC3": "GdFeO3-type distorted perovskite",
        "A2BC4": "Olivine ((Mg,Fe)2SiO4)",
        "ABC4": "Barite (BaSO₄)"
    },
    63: {  # Cmcm
        "A": "α-U structure",
        "AB": "CrB structure",
        "AB2": "HgBr2 type"
    },
    74: {  # Imma
        "AB": "TlI structure",
    },
    64: {  # Cmca
        "A": "α-Ga structure"
    },
    65: {  # Cmmm
        "A2B": "η-Fe2C structure"
    },
    70: {  # Fddd
        "A": "Orthorhombic unit cell"
    },

    # Monoclinic Structures
    14: {  # P21/c
        "AB": "Monoclinic structure",
        "AB2": "Baddeleyite (ZrO2)",
        "ABC4": "Monazite (CePO4)"
    },
    12: {  # C2/m
        "A2B2C7": "Thortveitite (Sc2Si2O7)"
    },
    15: {  # C2/c
        "A1B4C1D6": "Gypsum (CaH4O6S)",
        "ABC6": "Gypsum (CaH4O6S)",
        "ABC4": "Scheelite (CaWO₄)",
        "ABC5": "Sphene (CaTiSiO₅)"
    },
    1: {
        "A2B2C4D9": "Kaolinite"
    },
    # Triclinic Structures
    2: {  # P-1
        "AB": "Triclinic structure",
        "ABC3": "Wollastonite (CaSiO3)",
    },

    # Other important structures
    99: {  # P4mm
        "ABCD3": "Tetragonal perovskite"
    },
    167: {  # R-3c
        "ABC3": "Calcite (CaCO3)",
        "A2B3": "Corundum (Al2O3)"
    },
    176: {  # P6_3/m
        "A10B6C2D31E": "Apatite (Ca10(PO4)6(OH)2)",
        "A5B3C1D13": "Apatite (Ca5(PO4)3OH",
        "A5B3C13": "Apatite (Ca5(PO4)3OH"
    },
    58: {  # Pnnm
        "AB2": "Marcasite (FeS2)"
    },
    11: {  # P21/m
        "A2B": "ThSi2 type"
    },
    72: {  # Ibam
        "AB2": "MoSi2 type"
    },
    198: {  # P213
        "AB": "FeSi structure",
        "A12": "β-Mn structure"
    },
    88: {  # I41/a
        "ABC4": "Scheelite (CaWO4)"
    },
    33: {  # Pna21
        "AB": "FeAs structure"
    },
    130: {  # P4/ncc
        "AB2": "Cristobalite (SiO2)"
    },
    152: {  # P3121
        "AB2": "Quartz (SiO2)"
    },
    200: {  # Pm-3
        "A3B3C": "Fe3W3C"
    },
    224: {  # Pn-3m
        "AB": "Pyrochlore-related",
        "A2B": "Cuprite (Cu2O)"
    },
    127: {  # P4/mbm
        "AB": "σ-phase structure",
        "AB5": "CaCu5 type"
    },
    148: {  # R-3
        "ABC3": "Calcite (CaCO₃)",
        "ABC3": "Ilmenite (FeTiO₃)",
        "ABCD3": "Dolomite",
    },
    69: {  # Fmmm
        "A": "β-W structure"
    },
    128: {  # P4/mnc
        "A3B": "Cr3Si (A15)"
    },
    206: {  # Ia-3
        "AB2": "Pyrite derivative",
        "AB2": "Pyrochlore (defective)",
        "A2B3": "Bixbyite"
    },
    212: {  # P4_3 32

        "A4B3": "Mn4Si3 type"
    },
    180: {
        "AB2": "β-quartz (SiO2)",
    },
    226: {  # Fm-3c
        "AB2": "BiF3 type"
    },
    196: {  # F23
        "AB2": "FeS2 type"
    },
    96: {
        "AB2": "α-Cristobalite (SiO2)"
    }

}

def get_full_conventional_structure(structure, symprec=1e-3):
    # Create the spglib cell tuple: (lattice, fractional coords, atomic numbers)
    cell = (structure.lattice.matrix, structure.frac_coords,
            [max(site.species, key=site.species.get).number for site in structure])

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset['std_lattice']
    std_positions = dataset['std_positions']
    std_types = dataset['std_types']

    conv_structure = Structure(std_lattice, std_types, std_positions)
    return conv_structure

ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


MINERALS = {
    # Cubic structures
    225: {  # Fm-3m
        "Rock Salt (NaCl)": "Na Cl",
        "Fluorite (CaF2)": "Ca F2",
        "Anti-Fluorite (Li2O)": "Li2 O",
    },
    229: {  # Im-3m
        "BCC Iron": "Fe",
    },
    221: {  # Pm-3m
        "Perovskite (SrTiO3)": "Sr Ti O3",
        "ReO3 type": "Re O3",
        "Inverse-perovskite (Ca3TiN)": "Ca3 Ti N",
        "Cesium chloride (CsCl)": "Cs Cl"
    },
    227: {  # Fd-3m
        "Diamond": "C",

        "Normal spinel (MgAl2O4)": "Mg Al2 O4",
        "Inverse spinel (Fe3O4)": "Fe3 O4",
        "Pyrochlore (Ca2NbO7)": "Ca2 Nb2 O7",
        "β-Cristobalite (SiO2)": "Si O2"

    },
    216: {  # F-43m
        "Zinc Blende (ZnS)": "Zn S",
        "Half-anti-fluorite (Li4Ti)": "Li4 Ti"
    },
    215: {  # P-43m


    },
    230: {  # Ia-3d
        "Garnet (Ca3Al2Si3O12)": "Ca3 Al2 Si3 O12",
    },
    205: {  # Pa-3
        "Pyrite (FeS2)": "Fe S2",
    },
    224:{
        "Cuprite (Cu2O)": "Cu2 O",
    },
    # Hexagonal structures
    194: {  # P6_3/mmc
        "HCP Magnesium": "Mg",
        "Ni3Sn type": "Ni3 Sn",
        "Graphite": "C",
        "MoS2 type": "Mo S2",
        "Nickeline (NiAs)": "Ni As",
    },
    186: {  # P6_3mc
        "Wurtzite (ZnS)": "Zn S"
    },
    191: {  # P6/mmm


        "AlB2 type": "Al B2",
        "CaCu5 type": "Ca Cu5"
    },
    #187: {  # P-6m2
#
 #   },
    156: {
        "CdI2 type": "Cd I2",
    },
    164: {
    "CdI2 type": "Cd I2",
    },
    166: {  # R-3m
    "Delafossite (CuAlO2)": "Cu Al O2"
    },
    # Tetragonal structures
    139: {  # I4/mmm
        "β-Tin (Sn)": "Sn",
        "MoSi2 type": "Mo Si2"
    },
    136: {  # P4_2/mnm
        "Rutile (TiO2)": "Ti O2"
    },
    123: {  # P4/mmm
        "CuAu (L10)": "Cu Au"
    },
    141: {  # I41/amd
        "Anatase (TiO2)": "Ti O2",
        "Zircon (ZrSiO4)": "Zr Si O4"
    },
    122: {  # P-4m2
        "Chalcopyrite (CuFeS2)": "Cu Fe S2"
    },
    129: {  # P4/nmm
        "PbO structure": "Pb O"
    },

    # Orthorhombic structures
    62: {  # Pnma
        "Aragonite (CaCO3)": "Ca C O3",
        "Cotunnite (PbCl2)": "Pb Cl2",
        "Olivine (Mg2SiO4)": "Mg2 Si O4",
        "Barite (BaSO4)": "Ba S O4",
        "Perovskite (GdFeO3)": "Gd Fe O3"
    },
    63: {  # Cmcm
        "α-Uranium": "U",
        "CrB structure": "Cr B",
        "TlI structure": "Tl I",
    },
   # 74: {  # Imma
   #
   # },
    64: {  # Cmca
        "α-Gallium": "Ga"
    },

    # Monoclinic structures
    14: {  # P21/c
        "Baddeleyite (ZrO2)": "Zr O2",
        "Monazite (CePO4)": "Ce P O4"
    },
    206: {  # C2/m
        "Bixbyite (Mn2O3)": "Mn2 O3"
    },
    15: {  # C2/c
        "Gypsum (CaSO4·2H2O)": "Ca S H4 O6",
        "Scheelite (CaWO4)": "Ca W O4"
    },

    1: {
        "Kaolinite": "Al2 Si2 O9 H4"

    },
    # Triclinic structures
    2: {  # P-1
        "Wollastonite (CaSiO3)": "Ca Si O3",
        #"Kaolinite": "Al2 Si2 O5"
    },

    # Other important structures
    167: {  # R-3c
        "Calcite (CaCO3)": "Ca C O3",
        "Corundum (Al2O3)": "Al2 O3"
    },
    176: {  # P6_3/m
        "Apatite (Ca5(PO4)3OH)": "Ca5 P3 O13 H"
    },
    58: {  # Pnnm
        "Marcasite (FeS2)": "Fe S2"
    },
    198: {  # P213
        "FeSi structure": "Fe Si"
    },
    88: {  # I41/a
        "Scheelite (CaWO4)": "Ca W O4"
    },
    33: {  # Pna21
        "FeAs structure": "Fe As"
    },
    96: {  # P4/ncc
        "α-Cristobalite (SiO2)": "Si O2"
    },
    92: {
        "α-Cristobalite (SiO2)": "Si O2"
    },
    152: {  # P3121
        "Quartz (SiO2)": "Si O2"
    },
    148: {  # R-3
        "Ilmenite (FeTiO3)": "Fe Ti O3",
        "Dolomite (CaMgC2O6)": "Ca Mg C2 O6",
    },
    180: {  # P4_3 32
        "β-quartz (SiO2)": "Si O2"
    }
}


def get_cod_entries(params):
    try:
        response = requests.get('https://www.crystallography.net/cod/result', params=params)
        if response.status_code == 200:
            results = response.json()
            return results  # Returns a list of entries
        else:
            st.error(f"COD search error: {response.status_code}")
            return []
    except Exception as e:
        st.write(
            "Error during connection to COD database. Probably reason is that the COD database server is currently down.")


def get_cif_from_cod(entry):
    file_url = entry.get('file')
    if file_url:
        response = requests.get(f"https://www.crystallography.net/cod/{file_url}.cif")
        if response.status_code == 200:
            return response.text
    return None


def get_structure_from_mp(mp_id):
    with MPRester(MP_API_KEY) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)
        return structure


from pymatgen.io.cif import CifParser


def get_structure_from_cif_url(cif_url):
    response = requests.get(f"https://www.crystallography.net/cod/{cif_url}.cif")
    if response.status_code == 200:
        #  writer = CifWriter(response.text, symprec=0.01)
        #  parser = CifParser.from_string(writer)
        #  structure = parser.get_structures(primitive=False)[0]
        return response.text
    else:
        raise ValueError(f"Failed to fetch CIF from URL: {cif_url}")


def get_cod_str(cif_content):
    parser = CifParser.from_str(cif_content)
    structure = parser.get_structures(primitive=False)[0]
    return structure

def sort_formula_alphabetically(formula_input):
    formula_parts = formula_input.strip().split()
    return " ".join(sorted(formula_parts))



def fetch_and_parse_cod_cif(entry):
    file_id = entry.get('file')
    if not file_id:
        return None, None, None, "Missing file ID in entry"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        cif_url = f"https://www.crystallography.net/cod/{file_id}.cif"
        response = requests.get(cif_url, timeout=15, headers=headers)
        response.raise_for_status()
        cif_content = response.text
        parser = CifParser.from_str(cif_content)

        structure = parser.get_structures(primitive=False)[0]
        cod_id = f"cod_{file_id}"
        return cod_id, structure, entry, None

    except Exception as e:
        return None, None, None, str(e)


def get_all_sites(structure):
    #
    try:
        sga = SpacegroupAnalyzer(structure)
        sym_data = sga.get_symmetry_dataset()
        wyckoffs = sym_data.get("wyckoffs", ["?"] * len(structure))
    except Exception:
        wyckoffs = ["?"] * len(structure)

    all_sites = []
    for i, site in enumerate(structure):

        if site.is_ordered:
            element = site.specie.symbol
        else:
            element = ", ".join(f"{sp.symbol}:{occ:.3f}" for sp, occ in site.species.items())

        all_sites.append({
            "site_index": i,
            "wyckoff_letter": wyckoffs[i],
            "element": element,
            "coords": site.frac_coords
        })

    return all_sites
def get_unique_sites(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        symmetry_data = analyzer.get_symmetry_dataset()
        wyckoff_letters = symmetry_data["wyckoffs"]
        equivalent_sites = analyzer.get_symmetrized_structure().equivalent_sites
        equivalent_indices = analyzer.get_symmetrized_structure().equivalent_indices

        unique_sites = []
        for i, equiv_indices in enumerate(equivalent_indices):
            site_index = equiv_indices[0]
            site = structure[site_index]

            if site.is_ordered:
                element = site.specie.symbol
            else:
                element = ", ".join([f"{sp.symbol}: {occ:.3f}" for sp, occ in site.species.items()])

            wyckoff = wyckoff_letters[site_index]
            coords = site.frac_coords
            unique_sites.append({
                'wyckoff_index': i,
                'site_index': site_index,
                'wyckoff_letter': wyckoff,
                'element': element,
                'coords': coords,
                'multiplicity': len(equiv_indices),
                'equivalent_indices': equiv_indices
            })

        return unique_sites
    except Exception as e:
        unique_sites = []
        for i, site in enumerate(structure):
            if site.is_ordered:
                element = site.specie.symbol
            else:
                element = ", ".join([f"{sp.symbol}: {occ:.3f}" for sp, occ in site.species.items()])

            unique_sites.append({
                'wyckoff_index': i,
                'site_index': i,
                'wyckoff_letter': "?",
                'element': element,
                'coords': site.frac_coords,
                'multiplicity': 1,
                'equivalent_indices': [i]
            })

        return unique_sites
