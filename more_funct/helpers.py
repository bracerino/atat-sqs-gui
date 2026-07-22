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
def extract_space_group_number(selected_option):
    if selected_option:
        return int(selected_option.split(' ')[0])
    return None


SPACE_GROUP_OPTIONS = [f"{num} ({symbol})" for num, symbol in SPACE_GROUP_SYMBOLS.items()]




from pymatgen.ext.optimade import OptimadeRester

def search_mc3d_optimade(query_params, limit=300):
    import requests
    from pymatgen.core import Structure, Lattice, Composition
    import streamlit as st

    # st.write("=" * 50)
    # st.write("🔍 **MC3D SEARCH DEBUG INFO**")
    # st.write(f"📝 Query parameters: {query_params}")
    # st.write(f"🔢 Limit: {limit}")

    endpoints = [
        "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures",
        "https://optimade.materialscloud.org/main/mc3d-pbe-v1/v1/structures",
    ]

    filter_parts = []
    strict_elements = None
    target_composition = None
    check_composition = False

    if 'elements' in query_params:
        elements = query_params['elements']
        strict_elements = set(elements)
        # st.write(f"🧪 Searching for elements: {elements} (STRICT - only these elements)")
        for el in elements:
            filter_parts.append(f'elements HAS "{el}"')

    if 'formula' in query_params:
        formula_input = query_params['formula'].replace(' ', '')
        # st.write(f"⚗️ Formula input from user: {formula_input}")

        try:
            target_composition = Composition(formula_input)
            check_composition = True
            elements_from_formula = sorted([str(el) for el in target_composition.elements])

            # st.write(f"✨ User's formula: {formula_input}")
            # st.write(f"✨ Pymatgen reduced formula: {target_composition.reduced_formula}")
            # st.write(f"✨ MC3D likely stores as: {''.join([el + str(int(target_composition[el])) if target_composition[el] != 1 else el for el in elements_from_formula])}")
            # st.write(f"🧪 Searching by elements: {elements_from_formula} (order-independent)")

            for el in elements_from_formula:
                filter_parts.append(f'elements HAS "{el}"')

            strict_elements = set(elements_from_formula)

        except Exception as e:
            st.warning(f"⚠️ Could not parse formula: {e}")

    filter_str = " AND ".join(filter_parts) if filter_parts else None

    params = {
        'page_limit': min(limit, 100)
    }
    if filter_str:
        params['filter'] = filter_str
        # st.write(f"🔍 **OPTIMADE Filter**: `{filter_str}`")
    # else:
    #     st.write("⚠️ No filter applied - fetching first results")

    for idx, endpoint in enumerate(endpoints):
        # st.write("-" * 50)
        # st.write(f"🌐 **Endpoint {idx + 1}/{len(endpoints)}**: {endpoint}")

        try:
            # st.write(f"📤 Sending request with params: {params}")
            response = requests.get(endpoint, params=params, timeout=30)
            # st.write(f"📡 **Response Status Code**: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # st.write(f"📦 Response keys: {list(data.keys())}")

                entries = data.get('data', [])
                # st.write(f"📊 **Number of entries in response**: {len(entries)}")

                if not entries:
                    # st.warning(f"⚠️ No entries returned from this endpoint")
                    # if 'meta' in data:
                    #     st.write(f"ℹ️ Meta info: {data['meta']}")
                    continue

                # if entries:
                #     st.write(f"🔬 First entry ID: {entries[0].get('id', 'N/A')}")
                #     st.write(f"🔬 First entry attributes keys: {list(entries[0].get('attributes', {}).keys())}")

                structures = []
                parse_errors = 0
                filtered_out = 0

                for entry_idx, entry in enumerate(entries[:limit]):
                    try:
                        attrs = entry['attributes']
                        entry_id = attrs.get('_mcloud_mc3d_id', entry['id'])

                        lattice_vectors = attrs.get('lattice_vectors')
                        if not lattice_vectors:
                            parse_errors += 1
                            continue

                        lattice = Lattice(lattice_vectors)

                        species = attrs.get('species_at_sites', [])
                        if not species:
                            parse_errors += 1
                            continue

                        if 'cartesian_site_positions' in attrs:
                            coords = attrs['cartesian_site_positions']
                            coords_are_cartesian = True
                        elif 'fractional_site_positions' in attrs:
                            coords = attrs['fractional_site_positions']
                            coords_are_cartesian = False
                        else:
                            parse_errors += 1
                            continue

                        structure = Structure(
                            lattice,
                            species,
                            coords,
                            coords_are_cartesian=coords_are_cartesian
                        )

                        if strict_elements:
                            structure_elements = set([str(el) for el in structure.composition.elements])
                            if structure_elements != strict_elements:
                                filtered_out += 1
                                continue

                            if check_composition and target_composition:
                                structure_comp = structure.composition.reduced_composition
                                target_comp = target_composition.reduced_composition

                                # if entry_idx < 3:
                                #     st.write(f"🔬 Entry #{entry_idx + 1} - {entry_id}:")
                                #     st.write(f"   - Target composition: {target_comp}")
                                #     st.write(f"   - Structure composition: {structure_comp}")
                                #     st.write(f"   - Match: {structure_comp == target_comp}")

                                if structure_comp != target_comp:
                                    filtered_out += 1
                                    continue

                        formula = attrs.get('chemical_formula_reduced', structure.composition.reduced_formula)

                        structures.append({
                            'id': entry_id,
                            'structure': structure,
                            'formula': formula
                        })

                        # if (entry_idx + 1) % 10 == 0:
                        #     st.write(f"✅ Parsed {entry_idx + 1}/{len(entries[:limit])} entries...")

                    except Exception as e:
                        parse_errors += 1
                        continue

                # st.write(f"📈 **Parsing Summary**:")
                # st.write(f"   - Successfully parsed: {len(structures)}")
                # st.write(f"   - Failed to parse: {parse_errors}")
                # if strict_elements:
                #     st.write(f"   - Filtered out (wrong elements): {filtered_out}")

                if structures:
                   #st.success(f"✅ Found {len(structures)} structures in MC3D via OPTIMADE.")
                    # st.write("=" * 50)
                    return structures
                # else:
                #     st.error("❌ No structures could be parsed from this endpoint")

            # elif response.status_code == 404:
            #     st.warning(f"⚠️ Endpoint not found (404)")
            # elif response.status_code == 500:
            #     st.error(f"❌ Server error (500)")
            #     try:
            #         error_data = response.json()
            #         st.write(f"Error details: {error_data}")
            #     except:
            #         st.write(f"Raw response: {response.text[:500]}")
            # else:
            #     st.warning(f"⚠️ Unexpected status code: {response.status_code}")
            #     st.write(f"Response text: {response.text[:500]}")

        except requests.exceptions.Timeout:
            st.error(f"⏱️ Request timed out for {endpoint}")
        except requests.exceptions.ConnectionError:
            st.error(f"🔌 Connection error for {endpoint}")
        except Exception as e:
            st.error(f"❌ Unexpected error with endpoint {endpoint}: {str(e)}")
            import traceback
            st.write(f"Traceback: {traceback.format_exc()}")
            continue

    st.error("❌ Could not retrieve structures from any MC3D endpoint")
    return []


def get_mc3d_structure_by_id(mc3d_id):

    import requests
    from pymatgen.core import Structure, Lattice
    import streamlit as st

    endpoints = [
        "https://optimade.materialscloud.org/main/mc3d-pbesol-v2/v1/structures",
        "https://optimade.materialscloud.org/main/mc3d-pbe-v1/v1/structures",
    ]

    params = {
        'filter': f'_mcloud_mc3d_id="{mc3d_id}"'
    }

    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                entries = data.get('data', [])

                if not entries:
                    continue

                entry = entries[0]
                attrs = entry['attributes']

                lattice_vectors = attrs.get('lattice_vectors')
                if not lattice_vectors:
                    continue

                lattice = Lattice(lattice_vectors)
                species = attrs.get('species_at_sites', [])

                if not species:
                    continue

                if 'cartesian_site_positions' in attrs:
                    coords = attrs['cartesian_site_positions']
                    coords_are_cartesian = True
                elif 'fractional_site_positions' in attrs:
                    coords = attrs['fractional_site_positions']
                    coords_are_cartesian = False
                else:
                    continue

                structure = Structure(
                    lattice,
                    species,
                    coords,
                    coords_are_cartesian=coords_are_cartesian
                )

                return structure

        except Exception as e:
            continue

    st.warning(f"Could not fetch structure {mc3d_id} from MC3D")
    return None

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
    # Left column: the upload prompt. Middle column: pick an example crystal
    # structure (bcc / fcc / sc / hcp). Right column: one-click demo button that
    # loads a ready-made random alloy of that lattice and auto-fills every SQS
    # parameter so a new user can explore the full workflow.
    from more_funct.example_alloy import render_example_selector, render_example_alloy_button

    col_info, col_select, col_button = st.columns([2, 1.5, 1.5])
    with col_info:
        st.markdown("""
        <div style="
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid rgba(91, 140, 255, 0.35);
            background: linear-gradient(135deg,
                rgba(91, 140, 255, 0.10) 0%,
                rgba(91, 140, 255, 0.18) 100%);
            line-height: 1.55;">
            ⬅️ Please upload an initial <b>crystal structure</b> file (or search for it
            with the implemented interface within <b>MP, MC3D, or COD databases</b>)
            that will define the base atomic positions for SQS creation.
        </div>
        """, unsafe_allow_html=True)
    with col_select:
        render_example_selector()
    with col_button:
        render_example_alloy_button()

    st.markdown("#### Illustrative example:")

    with open("images_atat_gui/Uvodni_pro_aplikaci.png", "rb") as f:
        import base64
        data = f.read()
        encoded = base64.b64encode(data).decode()

    html_img = f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{encoded}" style="width:90%; max-width:800px; height:auto;">
    </div>
    """
    st.iframe(html_img, height=500)

    st.iframe("""
    <style>
    .glow-underline {
        width: 800px;              
        height: 4px;                 
        margin: 12px auto 24px auto; 
        border-radius: 999px;
        background: linear-gradient(90deg,
            rgba(91, 140, 255, 0) 0%,
            rgba(91, 140, 255, 0.8) 20%,
            rgba(91, 140, 255, 1) 50%,
            rgba(91, 140, 255, 0.8) 80%,
            rgba(91, 140, 255, 0) 100%);
        filter: blur(2px);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scaleX(0.95); opacity: 0.6; }
        50% { transform: scaleX(1.05); opacity: 1; }
    }
    </style>

    <div class="glow-underline"></div>
    """, height=40)
    
    

    st.markdown("""
### 🎥 Tutorials and References
- 📘 **General Overview of the App**: [Watch on YouTube](https://www.youtube.com/watch?v=GGo_9T5wqus)  
- ⚡ **Binary Alloy Workflow (Search Across Concentration Range)**: [Watch on YouTube](https://youtu.be/wL5re3Fu1nQ?si=4HGRmIQBX39zs-0B)
- 🎲 **How to create bcc HEA** [Watch on YouTube](https://youtu.be/U6JI2j3BfMg?si=Oeal6MueVzYRLlmc)
- 📖 [**Publication**](https://doi.org/10.1016/j.jocs.2026.102846): 
Read about the SimplySQS workflow and its application to the Pb<sub>1-x</sub>Sr<sub>x</sub>TiO<sub>3</sub> system. The study shows automated SQS generation across the entire concentration range using an all-in-one bash script, and evaluates the performance of a universal MLIP (MACE-MATPES-r2SCAN) for the cubic-to-tetragonal phase transition.


---

This tool provides a **graphical interface** for generating input files  
(`rndstr.in`, `sqscell.out`, `monitor.sh`) to create **Special Quasirandom Structures (SQS)** using the **ATAT mcsqs** package.  

It also enables **automated script generation** for binary alloys and batch execution of mcsqs searches across composition ranges.  

---
""",
               unsafe_allow_html=True)
    with open("images_atat_gui/Workflow.png", "rb") as f:
        import base64
        data = f.read()
        encoded = base64.b64encode(data).decode()

    html_img = f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{encoded}" style="width:90%; max-width:800px; height:auto;">
    </div>
    """
    st.iframe(html_img, height=800)

    st.iframe("""
    <style>
    .glow-underline {
        width: 800px;              
        height: 4px;                 
        margin: 12px auto 24px auto; 
        border-radius: 999px;
        background: linear-gradient(90deg,
            rgba(91, 140, 255, 0) 0%,
            rgba(91, 140, 255, 0.8) 20%,
            rgba(91, 140, 255, 1) 50%,
            rgba(91, 140, 255, 0.8) 80%,
            rgba(91, 140, 255, 0) 100%);
        filter: blur(2px);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scaleX(0.95); opacity: 0.6; }
        50% { transform: scaleX(1.05); opacity: 1; }
    }
    </style>

    <div class="glow-underline"></div>
    """, height=40)

    st.markdown("""
    ### ✨ Key Features

#### 🔬 Crystal Structures
- Upload custom crystal structures (CIF, POSCAR, LMP, XYZ with lattice)
- Retrieve directly from **Materials Project (MP)**, **AFLOW**, or **COD** databases  

---

#### 🎯 Supercell & Concentrations
- Define **supercell size / number of atoms**
- Set target **sublattice concentrations**  
- 📊 Automatically recalculate feasible concentrations  

---

#### 💾 Input File Generation
- Automatically generate **`rndstr.in`** and **`sqscell.out`** files   

---

#### 🛠️ Automated Workflows
- Create **bash scripts** to:  
  - Build input files (**`rndstr.in`**, **`sqscell.out`**)
  - Run **mcsqs** with optional parallelization  
  - Monitor convergence in real-time  
  - Convert automatically bestsqs.out into POSCAR
- Automate full SQS searches **across binary alloy composition ranges** (e.g. Ba₁₋ₓSrₓTiO₃)

---

#### 📤 File Handling
- Convert **`bestsqs.out`** into:  
  - POSCAR | LMP | CIF | XYZ  
- Upload log files (`mcsqs.log`, `mcsqs1.log`, …, `mcsqs_progress.csv`) for:  
  - Convergence analysis  
  - Best run detection  

---

#### 📈 Structure Analysis
- Compute the **Pair Radial Distribution Function (PRDF)** for `bestsqs.out`  

---

#### 🧹 Vacancies Creation
- Introduce **ordered vacancies** by selectively removing elements from SQS  

---

🚀 This tool streamlines the **entire SQS workflow** — from structure input to automated search, analysis, and output conversion.
     
     """,
               unsafe_allow_html=True)



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
        st.iframe(html_string, height=420, width=420)

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
        st.iframe(html_string, height=420, width=620)

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


import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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


from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.structure import PartialRadialDistributionFunction
from itertools import combinations
from collections import defaultdict
import plotly.graph_objects as go


from io import StringIO
from ase.constraints import FixAtoms


def pymatgen_to_ase(structure):
    from ase import Atoms
    import numpy as np

    symbols = []
    for site in structure:
        if site.is_ordered:
            # .symbol strips oxidation state (e.g. 'Mo4+' → 'Mo')
            symbols.append(site.specie.symbol)
        else:
            dominant = max(site.species, key=lambda sp: site.species[sp])
            symbols.append(dominant.symbol)

    positions = [site.coords for site in structure]
    cell = structure.lattice.matrix

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )
    return atoms



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
       # print(formula)
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
    cell = (structure.lattice.matrix, structure.frac_coords,
            [max(site.species, key=site.species.get).number for site in structure])

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    std_lattice = dataset.std_lattice
    std_positions = dataset.std_positions
    std_types = dataset.std_types

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


from pymatgen.io.cif import CifParser


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

        structure = parser.parse_structures(primitive=False)[0]
        cod_id = f"cod_{file_id}"
        return cod_id, structure, entry, None

    except Exception as e:
        return None, None, None, str(e)


def get_all_sites(structure):
    #
    try:
        sga = SpacegroupAnalyzer(structure)
        sym_data = sga.get_symmetry_dataset()
        wyckoffs = sym_data.wyckoffs if sym_data else ["?"] * len(structure)
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
        #wyckoff_letters = symmetry_data["wyckoffs"]
        wyckoff_letters = symmetry_data.wyckoffs
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
