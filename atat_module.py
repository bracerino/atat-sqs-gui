import streamlit as st
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.build import make_supercell
from helpers import *
from parallel_analysis import *


def render_concentration_sweep_section(chemical_symbols, target_concentrations, transformation_matrix,
                                       primitive_structure, cutoffs, total_atoms_in_supercell):
    if not target_concentrations:
        return

    is_binary = False
    sweep_element = None
    complement_element = None
    selected_sublattice = None

    if isinstance(target_concentrations, dict):
        first_value = next(iter(target_concentrations.values()))

        if isinstance(first_value, dict):
            binary_sublattices = []
            for sublattice_letter, concentrations in target_concentrations.items():
                if len(concentrations) == 2:
                    elements = list(concentrations.keys())
                    binary_sublattices.append((sublattice_letter, elements))

            if len(binary_sublattices) == 0:
                return
            elif len(binary_sublattices) == 1:
                is_binary = True
                selected_sublattice, elements = binary_sublattices[0]
                sweep_element = elements[0]
                complement_element = elements[1]
            else:
                st.markdown("---")
                st.subheader("🔄 Concentration Sweep Mode")
                st.info("Multiple binary sublattices detected. Please select one for concentration sweep:")
                st.info(f"You can generate a script to automatically run the mcsqs search across concentration range.")

                sublattice_options = []
                for sublattice_letter, elements in binary_sublattices:
                    sublattice_options.append(f"Sublattice {sublattice_letter}: {elements[0]} + {elements[1]}")

                selected_option = st.selectbox(
                    "Choose sublattice for concentration sweep:",
                    options=sublattice_options,
                    index=0
                )

                if selected_option:
                    selected_idx = sublattice_options.index(selected_option)
                    selected_sublattice, elements = binary_sublattices[selected_idx]
                    is_binary = True
                    sweep_element = elements[0]
                    complement_element = elements[1]
                else:
                    return
        else:
            if len(target_concentrations) == 2:
                is_binary = True
                elements = list(target_concentrations.keys())
                sweep_element = elements[0]
                complement_element = elements[1]

    if not is_binary:
        return

    st.markdown("---")
    st.subheader("🔄 Concentration Sweep Mode")

    if selected_sublattice:
        st.info(f"**Binary sublattice {selected_sublattice} detected:** {sweep_element} + {complement_element}")
        st.info(f"You can generate a script to automatically run the mcsqs search across concentration range.")
    else:
        st.info(f"**Binary system detected:** {sweep_element} + {complement_element}")
        st.info(f"You can generate a script to automatically run the mcsqs search across concentration range.")

    enable_sweep = st.checkbox(
        f"Enable concentration sweep for {sweep_element}",
        value=False,
        help=f"Generate bash script to automatically test multiple achievable concentrations of {sweep_element}"
    )

    if not enable_sweep:
        return

    supercell_factor = calculate_supercell_factor(transformation_matrix)

    if selected_sublattice:
        sublattice_sites = 0
        sublattice_elements = set([sweep_element, complement_element])

        for i, site_elements in enumerate(chemical_symbols):
            if isinstance(site_elements, list) and len(site_elements) > 1:
                if set(site_elements) == sublattice_elements:
                    sublattice_sites += 1

        total_sites_for_sublattice = sublattice_sites * supercell_factor
        st.write(f"**Sites for sublattice {selected_sublattice} in initial unit cell:** {sublattice_sites}")
        st.write(f"**Total sites for sublattice {selected_sublattice} in supercell:** {total_sites_for_sublattice}")

        possible_concentrations = []
        for i in range(1, total_sites_for_sublattice):
            conc = i / total_sites_for_sublattice
            possible_concentrations.append(round(conc, 6))

    else:
        st.write(f"**Supercell multiplicity:** {supercell_factor}")
        st.write(f"**Minimum concentration step:** 1/{supercell_factor} = {1 / supercell_factor:.6f}")
        st.info("**Global Mode:** Each concentration applies to ALL atomic sites equally")

        possible_concentrations = []
        for i in range(1, supercell_factor):
            conc = i / supercell_factor
            possible_concentrations.append(round(conc, 6))

        atoms = pymatgen_to_ase(primitive_structure)
        total_sites_for_sublattice = len(atoms) * supercell_factor

    col1, col2 = st.columns(2)

    with col1:
        if selected_sublattice:
            st.write(
                f"**Available {sweep_element} concentrations for sublattice {selected_sublattice}:** {len(possible_concentrations)}")
        else:
            st.write(f"**Available {sweep_element} concentrations:** {len(possible_concentrations)}")
            st.write("**Valid concentrations:** " + ", ".join([f"{c:.3f}" for c in possible_concentrations]))

        st.write("**Concentration Sampling:**")
        col_sample1, col_sample2 = st.columns(2)

        with col_sample1:
            sample_every_nth = st.number_input(
                "Sample every nth concentration:",
                min_value=1,
                max_value=len(possible_concentrations),
                value=1,
                step=1,
                help="Select every nth concentration (1 = all, 2 = every other, 3 = every third, etc.)",
                key="sample_every_nth"
            )

        with col_sample2:
            start_from = st.number_input(
                "Start from index:",
                min_value=0,
                max_value=len(possible_concentrations) - 1,
                value=0,
                step=1,
                help="Starting index for sampling (0 = first concentration)",
                key="start_from_index"
            )

        sampled_indices = list(range(start_from, len(possible_concentrations), sample_every_nth))
        default_concentrations = [possible_concentrations[i] for i in sampled_indices]

        if sample_every_nth > 1 or start_from > 0:
            st.info(
                f"📊 Sampling: {len(default_concentrations)} concentrations selected (every {sample_every_nth} starting from index {start_from})")

        selected_concentrations = st.multiselect(
            f"Select {sweep_element} concentrations for SQS generation:",
            options=possible_concentrations,
            default=default_concentrations,
            help="Choose from achievable concentrations" + (
                " based on sublattice size" if selected_sublattice else " (multiples of 1/supercell_multiplicity)")
        )

    with col2:
        mcsqs_mode = st.radio(
            "Choose MCSQS mode:",
            options=["Supercell Mode", "Atom Count Mode"],
            index=0,
            help="Supercell mode (-rc): Uses predefined supercell transformation\nAtom Count mode (-n): Searches for optimal supercell with specified atom count",
            key="concentration_sweep_mcsqs_mode"
        )

        if mcsqs_mode == "Supercell Mode":
            mode_description = "Uses -rc flag (predefined supercell)"
        else:
            mode_description = f"Uses -n flag (optimize for total of {total_atoms_in_supercell} atoms)"

        st.caption(mode_description)

        time_per_conc = st.number_input(
            "Time per concentration (minutes)",
            min_value=0.1,
            max_value=14400.0,
            value=30.0,
            step=0.5,
            help="How long to run mcsqs for each concentration"
        )

        max_parallel = st.number_input(
            "Maximum parallel jobs",
            min_value=1,
            max_value=200,
            value=4,
            step=1,
            help="Number of parallel mcsqs processes"
        )

        parallel_runs_per_conc = st.number_input(
            "Parallel runs per concentration",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
            help="Number of parallel mcsqs instances to run for each concentration"
        )

        progress_update_interval = st.number_input(
            "Progress update interval (seconds)",
            min_value=1,
            max_value=6000,
            value=10,
            step=1,
            help="How often to print the progress update to the console in seconds"
        )

    if not selected_concentrations:
        st.warning("Please select at least one concentration for SQS generation.")
        return

    st.write(f"**Selected concentrations:** {selected_concentrations}")
    st.write(f"**Total estimated time:** {len(selected_concentrations) * time_per_conc:.1f} minutes")

    if st.button("Generate Concentration Sweep Script", type="primary"):
        if selected_sublattice:
            filtered_target_concentrations = {
                selected_sublattice: target_concentrations[selected_sublattice]
            }
        else:
            filtered_target_concentrations = target_concentrations

        script_content = generate_concentration_sweep_script(
            sweep_element,
            complement_element,
            selected_concentrations,
            time_per_conc,
            max_parallel,
            parallel_runs_per_conc,
            filtered_target_concentrations,
            chemical_symbols,
            transformation_matrix,
            primitive_structure,
            cutoffs,
            total_sites_for_sublattice,
            progress_update_interval,
            mcsqs_mode, total_atoms_in_supercell
        )

        st.download_button(
            label="📥 Download Concentration Sweep Script",
            data=script_content,
            file_name="concentration_sweep.sh",
            mime="text/plain",
            type="primary"
        )

        st.success(
            "Concentration sweep script generated! Make sure to have these prerequisites: **`ATAT mcsqs`**, **`NumPy module in Python`**")

        with st.expander("Script Preview", expanded=False):
            st.code(script_content, language="bash")


def generate_concentration_sweep_script(sweep_element, complement_element, selected_concentrations,
                                        time_per_conc, max_parallel, parallel_runs_per_conc,
                                        target_concentrations, chemical_symbols, transformation_matrix,
                                        primitive_structure, cutoffs, total_sites, progress_update_interval, mcsqs_mode,
                                        total_atoms_in_supercell):
    corrdump_cmd = "corrdump -l=rndstr.in -ro -noe -nop -clus"
    if len(cutoffs) >= 1 and cutoffs[0] is not None:
        corrdump_cmd += f" -2={cutoffs[0]}"
    if len(cutoffs) >= 2 and cutoffs[1] is not None:
        corrdump_cmd += f" -3={cutoffs[1]}"
    if len(cutoffs) >= 3 and cutoffs[2] is not None:
        corrdump_cmd += f" -4={cutoffs[2]}"

    atoms = pymatgen_to_ase(primitive_structure)
    original_lattice = primitive_structure.lattice
    max_param = max(original_lattice.a, original_lattice.b, original_lattice.c)
    if mcsqs_mode == "Supercell Mode":
        mcsqs_base_cmd = "mcsqs -rc"
        mode_description = "supercell mode (-rc)"
    else:
        mcsqs_base_cmd = f"mcsqs -n {total_atoms_in_supercell}"
        mode_description = f"atom count mode (-n {total_atoms_in_supercell})"
    script_lines = [
        "#!/bin/bash",
        "",
        f"# Concentration sweep script for {sweep_element}-{complement_element} system",
        f"# Generated by SimplySQS",
        f"# Corrdump command: {corrdump_cmd}",
        f"# Total sites in supercell: {total_sites}",
        f"# Parallel runs per concentration: {parallel_runs_per_conc}",
        f"# Maximum concurrent jobs: {max_parallel}",
        f"# Concentrations: {selected_concentrations}",
        f"# Total estimated time: {len(selected_concentrations) * time_per_conc} minutes",
        "",
        "set -e",
        "",
        f'SWEEP_ELEMENT="{sweep_element}"',
        f'COMPLEMENT_ELEMENT="{complement_element}"',
        f"TIME_PER_CONC_DEFAULT={time_per_conc}",
        f"MAX_PARALLEL={max_parallel}",
        f"PARALLEL_RUNS_PER_CONC_DEFAULT={parallel_runs_per_conc}",
        f'CORRDUMP_CMD="{corrdump_cmd}"',
        f'MCSQS_BASE_CMD="{mcsqs_base_cmd}"',
        f'MCSQS_MODE="{mcsqs_mode}"',
        f"PROGRESS_UPDATE_INTERVAL={progress_update_interval}",
        "",
        'TOTAL_TIME_SECONDS=$(echo "$TIME_PER_CONC_DEFAULT * 60" | bc | xargs printf "%.0f")',
        "",
        "GLOBAL_START_TIME=$(date +%s)",
        "declare -A CONC_START_TIMES",
        "declare -A CONC_BEST_SCORES",
        "declare -A CONC_BEST_RUNS",
        "",
        'echo "🚀 Starting concentration sweep..."',
        'echo "🔬 Sweep element: $SWEEP_ELEMENT"',
        'echo "🔬 Complement element: $COMPLEMENT_ELEMENT"',
        'echo "🧪 Corrdump command: $CORRDUMP_CMD"',
        'echo "⚙️ MCSQS mode: $MCSQS_MODE"',
        'echo "⚙️ MCSQS command: $MCSQS_BASE_CMD"',
        'echo "⏱️ Default time per concentration: $TIME_PER_CONC_DEFAULT minutes"',
        f'echo "⚙️ Default parallel runs per concentration: $PARALLEL_RUNS_PER_CONC_DEFAULT"',
        f'echo "🔍 Concentrations to be searched: {selected_concentrations}"',
        "",
        'mkdir -p best_poscars',
        'echo "📂 Created folder best_poscars to store the best structures from each concentration."',
        "",
        "cleanup() {",
        '    echo "🧹 Cleaning up background processes..."',
        '    pkill -9 -f "mcsqs" 2>/dev/null || true',
        '    pkill -9 -f "monitor_progress" 2>/dev/null || true',
        '    jobs -p | xargs -r kill -9 2>/dev/null || true',
        "    wait 2>/dev/null || true",
        '    echo "✅ Cleanup complete."',
        "}",
        "",
        "trap cleanup EXIT INT TERM SIGINT SIGTERM",
        "",
        "format_elapsed_time() {",
        "    local elapsed=$1",
        "    local days=$((elapsed / 86400))",
        "    local hours=$(((elapsed % 86400) / 3600))",
        "    local minutes=$(((elapsed % 3600) / 60))",
        "    local seconds=$((elapsed % 60))",
        "    printf '%02d:%02d:%02d:%02d' $days $hours $minutes $seconds",
        "}",
        "",
        "extract_latest_objective() {",
        '    grep "Objective_function=" "$1" | tail -1 | sed "s/.*= *//" 2>/dev/null || echo ""',
        "}",
        "",
        "extract_latest_step() {",
        '    grep -c "Objective_function=" "$1" 2>/dev/null || echo "0"',
        "}",
        "",
        "get_best_objective_and_run() {",
        "    local best_obj=\"N/A\"",
        "    local best_run=\"N/A\"",
        "    local parallel_runs_per_conc=$1",
        "    ",
        "    if [ $parallel_runs_per_conc -gt 1 ]; then",
        "        for ((i=1; i<=parallel_runs_per_conc; i++)); do",
        "            if [ -f \"mcsqs$i.log\" ]; then",
        "                local current_obj=$(extract_latest_objective \"mcsqs$i.log\")",
        "                if [ -n \"$current_obj\" ] && [ \"$current_obj\" != \"N/A\" ] && [ \"$current_obj\" != \"\" ]; then",
        "                    if [ \"$best_obj\" = \"N/A\" ] || awk \"BEGIN {exit !($current_obj < $best_obj)}\" 2>/dev/null; then",
        "                        best_obj=\"$current_obj\"",
        "                        best_run=\"$i\"",
        "                    fi",
        "                fi",
        "            fi",
        "        done",
        "    else",
        "        if [ -f \"mcsqs.log\" ]; then",
        "            local current_obj=$(extract_latest_objective \"mcsqs.log\")",
        "            if [ -n \"$current_obj\" ] && [ \"$current_obj\" != \"N/A\" ] && [ \"$current_obj\" != \"\" ]; then",
        "                best_obj=\"$current_obj\"",
        "                best_run=\"1\"",
        "            fi",
        "        fi",
        "    fi",
        "    ",
        "    echo \"$best_obj,$best_run\"",
        "}",
        "",
        "initialize_concentration_csv() {",
        "    local conc=$1",
        "    local parallel_runs_per_conc=$2",
        "    local csv_file=\"optimization_data_${conc}.csv\"",
        "    ",
        "    # Create CSV header",
        "    if [ $parallel_runs_per_conc -gt 1 ]; then",
        "        header=\"Minute,Timestamp,Best_Objective,Best_Run\"",
        "        for ((i=1; i<=parallel_runs_per_conc; i++)); do",
        "            header=\"$header,Run${i}_Steps,Run${i}_Objective,Run${i}_Status\"",
        "        done",
        "    else",
        "        header=\"Minute,Timestamp,Steps,Objective_Function,Status\"",
        "    fi",
        "    ",
        "    echo \"$header\" > \"$csv_file\"",
        "    echo \"$csv_file\"",
        "}",
        "",
        # NEW FUNCTION: Log data to CSV
        "log_to_csv() {",
        "    local csv_file=$1",
        "    local conc=$2",
        "    local parallel_runs_per_conc=$3",
        "    local elapsed_minutes=$4",
        "    ",
        "    local current_time=$(date +'%Y-%m-%d %H:%M:%S')",
        "    local result=$(get_best_objective_and_run $parallel_runs_per_conc)",
        "    local best_obj=$(echo $result | cut -d',' -f1)",
        "    local best_run=$(echo $result | cut -d',' -f2)",
        "    ",
        "    if [ $parallel_runs_per_conc -gt 1 ]; then",
        "        # Parallel runs: log all runs data",
        "        row_data=\"$elapsed_minutes,$current_time,$best_obj,$best_run\"",
        "        ",
        "        for ((i=1; i<=parallel_runs_per_conc; i++)); do",
        "            local log_file=\"mcsqs$i.log\"",
        "            local steps=\"0\"",
        "            local objective=\"N/A\"",
        "            local status=\"STOPPED\"",
        "            ",
        "            if pgrep -f \"mcsqs.*-ip=$i\" > /dev/null; then",
        "                status=\"RUNNING\"",
        "            fi",
        "            ",
        "            if [ -f \"$log_file\" ]; then",
        "                steps=$(extract_latest_step \"$log_file\")",
        "                objective=$(extract_latest_objective \"$log_file\")",
        "                steps=${steps:-\"0\"}",
        "                objective=${objective:-\"N/A\"}",
        "            fi",
        "            ",
        "            row_data=\"$row_data,$steps,$objective,$status\"",
        "        done",
        "    else",
        "        # Single run: log single run data",
        "        local steps=\"0\"",
        "        local objective=\"N/A\"",
        "        local status=\"STOPPED\"",
        "        ",
        "        if pgrep -f \"mcsqs\" > /dev/null; then",
        "            status=\"RUNNING\"",
        "        fi",
        "        ",
        "        if [ -f \"mcsqs.log\" ]; then",
        "            steps=$(extract_latest_step \"mcsqs.log\")",
        "            objective=$(extract_latest_objective \"mcsqs.log\")",
        "            steps=${steps:-\"0\"}",
        "            objective=${objective:-\"N/A\"}",
        "        fi",
        "        ",
        "        row_data=\"$elapsed_minutes,$current_time,$steps,$objective,$status\"",
        "    fi",
        "    ",
        "    echo \"$row_data\" >> \"$csv_file\"",
        "}",
        "",
        "convert_bestsqs_to_poscar() {",
        "    local bestsqs_file=$1",
        "    local poscar_file=$2",
        "    local conc=$3",
        "    ",
        "    if [ ! -f \"$bestsqs_file\" ]; then",
        '        echo "⚠️ Warning: $bestsqs_file not found"',
        "        return 1",
        "    fi",
        "    ",
        '    echo "🔄 Converting $bestsqs_file to $poscar_file..."',
        "    ",
        "    python3 << EOF",
        "import sys",
        "import numpy as np",
        "try:",
        "    def parse_bestsqs(filename):",
        "        with open(filename, 'r') as f:",
        "            lines = f.readlines()",
        "        ",
        "        A = np.array([[float(x) for x in lines[i].split()] for i in range(3)])",
        "        B = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])",
        "        ",
        f"        A_scaled = A * {max_param:.6f}",
        "        final_lattice = np.dot(B, A_scaled)",
        "        ",
        "        atoms = []",
        "        atoms = []",
        "        for i in range(6, len(lines)):",
        "            line = lines[i].strip()",
        "            if line:",
        "                parts = line.split()",
        "                if len(parts) >= 4:",
        "                    x, y, z, element = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]",
        "                    if element.lower() in ['vac', \"'vac\", 'vacancy', 'x']:",
        "                        continue",
        "                    cart_pos = np.dot([x, y, z], A_scaled)",
        "                    atoms.append((element, cart_pos))",
        "        ",
        "        return final_lattice, atoms",
        "",
        "    def write_poscar(lattice, atoms, filename, comment):",
        "        from collections import defaultdict",
        "        element_groups = defaultdict(list)",
        "        for element, pos in atoms:",
        "            element_groups[element].append(pos)",
        "        ",
        "        elements = sorted(element_groups.keys())",
        "        ",
        "        with open(filename, 'w') as f:",
        "            f.write(f'{comment}\\n')",
        "            f.write('1.0\\n')",
        "            ",
        "            for vec in lattice:",
        "                f.write(f'  {vec[0]:15.9f} {vec[1]:15.9f} {vec[2]:15.9f}\\n')",
        "            ",
        "            f.write(' '.join(elements) + '\\n')",
        "            f.write(' '.join(str(len(element_groups[el])) for el in elements) + '\\n')",
        "            ",
        "            f.write('Direct\\n')",
        "            inv_lattice = np.linalg.inv(lattice)",
        "            for element in elements:",
        "                for cart_pos in element_groups[element]:",
        "                    frac_pos = np.dot(cart_pos, inv_lattice)",
        "                    f.write(f'  {frac_pos[0]:15.9f} {frac_pos[1]:15.9f} {frac_pos[2]:15.9f}\\n')",
        "",
        f'    comment = "SQS {sweep_element}{complement_element} conc=$conc from $bestsqs_file"',
        f'    lattice, atoms = parse_bestsqs("$bestsqs_file")',
        f'    write_poscar(lattice, atoms, "$poscar_file", comment)',
        f'    print(f"Successfully converted $bestsqs_file to $poscar_file")',
        "except Exception as e:",
        '    print(f"Python script failed with error: {e}")',
        "    import traceback",
        "    traceback.print_exc()",
        "    sys.exit(1)",
        "EOF",
        "    ",
        "    local python_exit_code=$?",
        "    return $python_exit_code",
        "}",
        "",
        "monitor_progress() {",
        "    local conc=$1",
        "    local parallel_runs_per_conc=$2",
        "    local total_time_seconds=$3",
        "    local csv_file=$4",
        "    local elapsed_seconds=0",
        "    ",
        "    while [ $elapsed_seconds -lt $total_time_seconds ]; do",
        "        sleep $PROGRESS_UPDATE_INTERVAL",
        "        elapsed_seconds=$((elapsed_seconds + PROGRESS_UPDATE_INTERVAL))",
        "        ",
        "        local current_time=$(date +%s)",
        "        local global_elapsed=$((current_time - GLOBAL_START_TIME))",
        "        local conc_elapsed=$((current_time - CONC_START_TIMES[$conc]))",
        "        local elapsed_minutes=$((conc_elapsed / 60))",
        "        ",
        "        local result=$(get_best_objective_and_run $parallel_runs_per_conc)",
        "        local best_obj=$(echo $result | cut -d',' -f1)",
        "        local best_run=$(echo $result | cut -d',' -f2)",
        "        ",
        "        CONC_BEST_SCORES[$conc]=\"$best_obj\"",
        "        CONC_BEST_RUNS[$conc]=\"$best_run\"",
        "        ",
        "        log_to_csv \"$csv_file\" \"$conc\" \"$parallel_runs_per_conc\" \"$elapsed_minutes\"",
        "        ",
        "        local global_time_str=$(format_elapsed_time $global_elapsed)",
        "        local conc_time_str=$(format_elapsed_time $conc_elapsed)",
        "        ",
        "        if [ \"$best_run\" != \"N/A\" ] && [ $parallel_runs_per_conc -gt 1 ]; then",
        "            printf \"[Conc %s] [%s] Global: %s | Conc %s: %s (sec %d/%d) | Best obj: %s (run %s)\\n\" \\",
        "                   \"$conc\" \"$(date +'%H:%M:%S')\" \"$global_time_str\" \"$conc\" \"$conc_time_str\" \\",
        "                   \"$elapsed_seconds\" \"$total_time_seconds\" \"$best_obj\" \"$best_run\"",
        "        else",
        "            printf \"[Conc %s] [%s] Global: %s | Conc %s: %s (sec %d/%d) | Best obj: %s\\n\" \\",
        "                   \"$conc\" \"$(date +'%H:%M:%S')\" \"$global_time_str\" \"$conc\" \"$conc_time_str\" \\",
        "                   \"$elapsed_seconds\" \"$total_time_seconds\" \"$best_obj\"",
        "        fi",
        "    done",
        "}",
        "",
        "run_concentration() {",
        "    local conc=$1",
        "    local current_run=$2",
        "    local total_runs=$3",
        f'    local sweep_atoms=$(printf "%.0f" $(echo "$conc * {total_sites}" | bc))',
        f'    local comp_atoms=$(echo "{total_sites} - $sweep_atoms" | bc)',
        '    local folder="conc_${SWEEP_ELEMENT}_${conc}"',
        '    local comp_conc=$(echo "1.0 - $conc" | bc -l)',
        "    ",
        "    local time_per_conc_current=$TIME_PER_CONC_DEFAULT",
        "    local parallel_runs_per_conc_current=$PARALLEL_RUNS_PER_CONC_DEFAULT",
        "    ",
        "    if [ \"$sweep_atoms\" -le 1 ] || [ \"$comp_atoms\" -le 1 ]; then",
        "        time_per_conc_current=0.1",
        "        parallel_runs_per_conc_current=1",
        '        echo ""',
        '        echo "ℹ️  Note: Single or very few atoms detected. Reducing time to $time_per_conc_current min and parallel runs to $parallel_runs_per_conc_current."',
        "    fi",
        "    ",
        "    local total_time_seconds_current=$(echo \"$time_per_conc_current * 60\" | bc | xargs printf \"%.0f\")",
        "    ",
        "    CONC_START_TIMES[$conc]=$(date +%s)",
        "    ",
        '    echo ""',
        '    echo "=========================================="',
        '    echo "($current_run/$total_runs) 🔬 Starting concentration $conc for $SWEEP_ELEMENT"',
        '    ' 'echo "🔢 Target atoms: $sweep_atoms $SWEEP_ELEMENT + $comp_atoms $COMPLEMENT_ELEMENT"',
        '    echo "🏃 Running $parallel_runs_per_conc_current parallel instances for $time_per_conc_current minutes"',
        '    echo "=========================================="',
        '    mkdir -p "$folder"',
        '    cd "$folder"',
        "    ",
        "    local csv_file=$(initialize_concentration_csv \"$conc\" \"$parallel_runs_per_conc_current\")",
        '    echo "📊 CSV logging initialized: $csv_file"',
        "    ",
        "    cat > rndstr.in << EOF",
        f"{original_lattice.a / max_param:.6f} {original_lattice.b / max_param:.6f} {original_lattice.c / max_param:.6f} {original_lattice.alpha:.2f} {original_lattice.beta:.2f} {original_lattice.gamma:.2f}",
        "1 0 0",
        "0 1 0",
        "0 0 1"
    ]

    for i, site in enumerate(primitive_structure):
        coord_str = f"{site.frac_coords[0]:.6f} {site.frac_coords[1]:.6f} {site.frac_coords[2]:.6f}"

        # Check if chemical_symbols is None (global mode) or populated (sublattice mode)
        if chemical_symbols is None:
            # Global mode: all sites get the sweep elements
            script_lines.append(f"{coord_str} ${{SWEEP_ELEMENT}}=$conc,${{COMPLEMENT_ELEMENT}}=$comp_conc")
        else:
            # Sublattice mode: use the chemical_symbols array
            site_elements = chemical_symbols[i]

            if isinstance(site_elements, list) and len(site_elements) > 1:
                if set(site_elements) == {sweep_element, complement_element}:
                    script_lines.append(f"{coord_str} ${{SWEEP_ELEMENT}}=$conc,${{COMPLEMENT_ELEMENT}}=$comp_conc")
                else:
                    script_lines.append(f"{coord_str} {','.join(sorted(site_elements))}")
            else:
                element = site_elements[0] if isinstance(site_elements, list) else str(site.specie)
                script_lines.append(f"{coord_str} {element}")

    script_lines.extend([
        "EOF",
        "",
        "    cat > sqscell.out << EOF",
        f"1",
        f"",
        f"{transformation_matrix[0][0]} {transformation_matrix[0][1]} {transformation_matrix[0][2]}",
        f"{transformation_matrix[1][0]} {transformation_matrix[1][1]} {transformation_matrix[1][2]}",
        f"{transformation_matrix[2][0]} {transformation_matrix[2][1]} {transformation_matrix[2][2]}",
        "EOF",
        "",
        '    echo "✨ Generating clusters with corrdump..."',
        "    eval $CORRDUMP_CMD",
        "    if [ $? -ne 0 ]; then",
        '        echo "❌ ERROR: corrdump failed for concentration $conc"',
        "        cd ..",
        "        return 1",
        "    fi",
        "",
        '    echo "✨ Starting $parallel_runs_per_conc_current parallel mcsqs instances..."',
        "    ",
        "    local pids=()",
        "    if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        '        for ((i=1; i<=parallel_runs_per_conc_current; i++)); do',
        '            timeout ${total_time_seconds_current}s $MCSQS_BASE_CMD -ip=$i > mcsqs$i.log 2>&1 || true &',
        "            pids+=($!)",
        '            echo "  ✅ Started mcsqs run $i for concentration $conc (PID: $!)"',
        "        done",
        "    else",
        '        timeout ${total_time_seconds_current}s $MCSQS_BASE_CMD > mcsqs.log 2>&1 || true &',
        "        pids+=($!)",
        '            echo "  ✅ Started single mcsqs run for concentration $conc (PID: $!)"',
        "    fi",
        "    ",
        "    monitor_progress $conc $parallel_runs_per_conc_current $total_time_seconds_current \"$csv_file\" &",
        "    local monitor_pid=$!",
        "    ",
        "    for pid in \"${pids[@]}\"; do",
        "        wait $pid || true",
        "    done",
        "    ",
        "    kill $monitor_pid 2>/dev/null || true",
        "    wait $monitor_pid 2>/dev/null || true",
        "    ",
        "    local final_elapsed_minutes=$(echo \"scale=1; $time_per_conc_current\" | bc)",
        "    log_to_csv \"$csv_file\" \"$conc\" \"$parallel_runs_per_conc_current\" \"$final_elapsed_minutes\"",
        '    echo "📊 Final optimization data logged to $csv_file"',
        "    ",
        '    echo ""',
        '    echo "=========================================="',
        '    echo "📄 Processing results and converting to POSCAR format for concentration $conc..."',
        '    echo "=========================================="',
        "    ",
        "    declare -a successful_runs",
        "    declare -a run_scores",
        "    ",
        "    if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        "        for ((i=1; i<=parallel_runs_per_conc_current; i++)); do",
        "            if [ -f \"bestsqs$i.out\" ]; then",
        '                local score=$(extract_latest_objective "mcsqs$i.log")',
        "                if [ -n \"$score\" ] && [ \"$score\" != \"N/A\" ] && [ \"$score\" != \"\" ]; then",
        "                    successful_runs+=($i)",
        '                    run_scores+=("$score")',
        "                fi",
        "            fi",
        "        done",
        "    else",
        "        if [ -f \"bestsqs.out\" ]; then",
        '            local score=$(extract_latest_objective "mcsqs.log")',
        "            if [ -n \"$score\" ] && [ \"$score\" != \"N/A\" ] && [ \"$score\" != \"\" ]; then",
        "                successful_runs+=(1)",
        '                run_scores+=("$score")',
        "            fi",
        "        fi",
        "    fi",
        "    ",
        "    if [ ${#successful_runs[@]} -eq 0 ]; then",
        '        echo "❌ No successful runs for concentration $conc"',
        "        cd ..",
        "        return 1",
        "    fi",
        "    ",
        'echo "Found ${#successful_runs[@]} successful runs"',
        "    ",
        "    local sorted_indices=()",
        "    for ((i=0; i<${#successful_runs[@]}; i++)); do",
        "        sorted_indices+=($i)",
        "    done",
        "    ",
        "    for ((i=0; i<${#sorted_indices[@]}; i++)); do",
        "        for ((j=i+1; j<${#sorted_indices[@]}; j++)); do",
        "            local idx_i=${sorted_indices[i]}",
        "            local idx_j=${sorted_indices[j]}",
        "            local score_i=${run_scores[idx_i]}",
        "            local score_j=${run_scores[idx_j]}",
        '            if awk "BEGIN {exit !($score_j < $score_i)}" 2>/dev/null; then',
        "                local temp=${sorted_indices[i]}",
        "                sorted_indices[i]=${sorted_indices[j]}",
        "                sorted_indices[j]=$temp",
        "            fi",
        "        done",
        "    done",
        "    ",
        '    echo "Converting ${#successful_runs[@]} successful runs to POSCAR format:"',
        "    local best_run_found=false",
        "    local best_poscar_filename=\"\"",
        "    ",
        "    for ((rank=0; rank<${#sorted_indices[@]}; rank++)); do",
        "        local idx=${sorted_indices[rank]}",
        "        local run_num=${successful_runs[idx]}",
        "        local score=${run_scores[idx]}",
        "        ",
        "        local bestsqs_filename",
        "        local poscar_filename",
        "        if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        "            bestsqs_filename=\"bestsqs${run_num}.out\"",
        "            poscar_filename=\"POSCAR_$((rank + 1))\"",
        "        else",
        "            bestsqs_filename=\"bestsqs.out\"",
        "            poscar_filename=\"POSCAR\"",
        "        fi",
        "        ",
        "        if convert_bestsqs_to_poscar \"$bestsqs_filename\" \"$poscar_filename\" \"$conc\"; then",
        "            if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        '                echo "  ✔️ POSCAR_$((rank + 1)): Run $run_num (score: $score)"',
        "            else",
        '                echo "  ✔️ POSCAR: Run 1 (score: $score)"',
        "            fi",
        "            ",
        "            if [ $rank -eq 0 ]; then",
        "                best_run_found=true",
        "                best_poscar_filename=\"$poscar_filename\"",
        "                if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        "                    cp \"POSCAR_1\" \"POSCAR\"",
        '                    echo "  🏆 → Best result: POSCAR_1 (also saved as POSCAR)"',
        "                else",
        '                    echo "  🏆 → Best result: POSCAR"',
        "                fi",
        "            fi",
        "        else",
        "            if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        '                echo "  ❌ Failed to convert run $run_num to POSCAR"',
        "            else",
        '                echo "  ❌ Failed to convert single run to POSCAR"',
        "            fi",
        "        fi",
        "    done",
        "    ",
        "    if [ \"$best_run_found\" = true ]; then",
        "        local best_idx=${sorted_indices[0]}",
        "        local best_score=${run_scores[best_idx]}",
        '        echo ""',
        '        echo "✅ Concentration $conc completed successfully"',
        '        echo "✨ Best result has score: $best_score"',
        "        if [ $parallel_runs_per_conc_current -gt 1 ]; then",
        '            echo "📂 Generated ${#successful_runs[@]} POSCAR files (POSCAR_1 to POSCAR_${#successful_runs[@]})"',
        "        else",
        '            echo "📂 Generated POSCAR file"',
        "        fi",
        "        ",
        '        local file_prefix="$2"',
        '        cp "POSCAR" "../best_poscars/${file_prefix}_POSCAR-${conc}"',
        '        echo "📁 Copied best POSCAR to ../best_poscars/${file_prefix}_POSCAR-${conc}"',
        "        ",
        "    else",
        '        echo "❌ Failed to convert any results for concentration $conc"',
        "    fi",
        "    ",
        "    cd ..",
        "}",
        "",
        "export -f run_concentration",
        "export -f monitor_progress",
        "export -f format_elapsed_time",
        "export -f get_best_objective_and_run",
        "export -f convert_bestsqs_to_poscar",
        "export -f extract_latest_objective",
        "export -f extract_latest_objective",
        "export -f extract_latest_step",
        "export -f initialize_concentration_csv",
        "export -f log_to_csv",
        "",
        "concentrations=("
    ])

    for conc in selected_concentrations:
        script_lines.append(f"    {conc}")

    script_lines.extend([
        ")",
        "",
        'echo "Will process ${#concentrations[@]} concentrations with a default of $PARALLEL_RUNS_PER_CONC_DEFAULT parallel runs each"',
        'echo "Total estimated time: $(echo "${#concentrations[@]} * $TIME_PER_CONC_DEFAULT" | bc -l | xargs printf "%.1f") minutes"',
        'echo "Maximum concurrent jobs: $MAX_PARALLEL"',
        'echo ""',
        "",
        "total_concentrations=${#concentrations[@]}",
        'for ((i=0; i<total_concentrations; i++)); do',
        "    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do",
        "        sleep 5",
        "    done",
        "    ",
        '    run_concentration "${concentrations[i]}" "$((i+1))" "$total_concentrations" &',
        "done",
        "",
        "wait",
        "",
        'pkill -9 -f "mcsqs" 2>/dev/null || true',
        'sleep 2',
        "",
        'echo ""',
        'echo "========================================"',
        'echo "🏁 All concentrations completed!"',
        'echo "========================================"',
        "",
        'echo "📋 Summary of generated files:"',
        'echo "- 📁 Each concentration folder contains POSCAR_1, POSCAR_2, etc. (ordered by objective function)"',
        'echo "- 📁 A copy of each best POSCAR is saved in the best_poscars folder"',
        "",
        'echo "✅ Concentration sweep completed successfully!"'
    ])

    return '\n'.join(script_lines)


def calculate_first_six_nn_atat_aware(structure, chem_symbols=None, use_sublattice_mode=False):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    original_lattice = structure.lattice
    a, b, c = original_lattice.abc
    max_param = max(a, b, c)

    from pymatgen.core.lattice import Lattice
    normalized_lattice = Lattice.from_parameters(
        a / max_param, b / max_param, c / max_param,
        original_lattice.alpha, original_lattice.beta, original_lattice.gamma
    )

    normalized_structure = structure.copy()
    normalized_structure.lattice = normalized_lattice
    sga = SpacegroupAnalyzer(normalized_structure)
    wyckoff_symbols = sga.get_symmetry_dataset().wyckoffs

    active_sites = []
    if use_sublattice_mode and chem_symbols:
        mixed_occupancy_sites = []
        for i, site_elements in enumerate(chem_symbols):
            if len(site_elements) >= 2:
                mixed_occupancy_sites.append(i)

        if not mixed_occupancy_sites:
            return {
                'overall': [],
                'message': "No mixed-occupancy sites found. ATAT requires at least 2 elements per site for cluster calculations."
            }
        wyckoff_to_sites = {}
        for i, wyckoff_symbol in enumerate(wyckoff_symbols):
            if wyckoff_symbol not in wyckoff_to_sites:
                wyckoff_to_sites[wyckoff_symbol] = []
            wyckoff_to_sites[wyckoff_symbol].append(i)

        wyckoff_positions_processed = set()
        for site_idx in mixed_occupancy_sites:
            wyckoff_symbol = wyckoff_symbols[site_idx]

            if wyckoff_symbol not in wyckoff_positions_processed:
                sites_with_same_wyckoff = wyckoff_to_sites[wyckoff_symbol]

                mixed_sites_with_same_wyckoff = []
                for equiv_site in sites_with_same_wyckoff:
                    if equiv_site in mixed_occupancy_sites:
                        mixed_sites_with_same_wyckoff.append(equiv_site)

                for equiv_site in mixed_sites_with_same_wyckoff:
                    if equiv_site not in active_sites:
                        active_sites.append(equiv_site)

                wyckoff_positions_processed.add(wyckoff_symbol)

        if not active_sites:
            return {
                'overall': [],
                'message': "No active sites found after Wyckoff analysis."
            }
    else:
        active_sites = list(range(len(normalized_structure.sites)))

    for i, site_idx in enumerate(active_sites):
        site = normalized_structure[site_idx]

    active_lattice = normalized_structure.lattice
    active_species = []
    active_coords = []

    for site_idx in active_sites:
        site = normalized_structure[site_idx]
        active_species.append(site.specie)
        active_coords.append(site.frac_coords)

    if not active_species:
        return {'overall': [], 'message': "No active sites found."}

    from pymatgen.core import Structure
    active_structure = Structure(
        lattice=active_lattice,
        species=active_species,
        coords=active_coords,
        coords_are_cartesian=False
    )

    active_supercell = active_structure * (3, 3, 3)

    original_active_sites = len(active_structure)
    center_cell_start = 13 * original_active_sites
    center_cell_end = center_cell_start + original_active_sites

    overall_distances = []

    for i in range(center_cell_start, center_cell_end):
        center_site = active_supercell[i]

        for j, target_site in enumerate(active_supercell):
            if i == j:
                continue

            distance = center_site.distance(target_site)
            if distance > 0.001:
                overall_distances.append(distance)

    original_distances = overall_distances.copy()

    base_distances = []
    for d in sorted(set(overall_distances)):
        rounded_d = round(d, 3)
        if abs(rounded_d - round(rounded_d)) < 0.01:
            base_distances.append(rounded_d)

    for base_d in base_distances:
        if base_d > 0:
            scaled_2x = base_d * 2.0
            overall_distances.extend([scaled_2x] * 12)
            scaled_3x = base_d * 3.0
            overall_distances.extend([scaled_3x] * 6)

    if overall_distances:
        sorted_distances = sorted(overall_distances)

        distances_around_2 = [d for d in sorted_distances if 1.8 < d < 2.2]

        unique_distances = []
        for d in sorted_distances:
            rounded_d = round(d, 4)
            if rounded_d not in unique_distances:
                unique_distances.append(rounded_d)

    overall_shells = group_distances_into_shells(overall_distances) if overall_distances else []

    return {
        'overall': overall_shells[:6],
        'active_sites': active_sites,
        'total_sites': len(normalized_structure.sites)
    }


def group_distances_into_shells(distances_data, tolerance=0.001):  # Much smaller tolerance
    if not distances_data:
        return []

    distances_data.sort()
    shells = []
    shell_number = 1

    current_group = [distances_data[0]]

    for i in range(1, len(distances_data)):
        diff = abs(distances_data[i] - current_group[0])

        if diff <= tolerance:
            current_group.append(distances_data[i])
        else:
            avg_distance = sum(current_group) / len(current_group)
            shells.append({
                'shell': shell_number,
                'distance': avg_distance,
                'count': len(current_group)
            })
            shell_number += 1
            current_group = [distances_data[i]]

    if current_group:
        avg_distance = sum(current_group) / len(current_group)
        shells.append({
            'shell': shell_number,
            'distance': avg_distance,
            'count': len(current_group)
        })

    return shells


def calculate_sqs_prdf(structure, cutoff=10.0, bin_size=0.1):
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        from matminer.featurizers.structure import PartialRadialDistributionFunction
        from itertools import combinations
        from collections import defaultdict

        elements = list(set([site.specie.symbol for site in structure if site.is_ordered]))

        species_combinations = list(combinations(elements, 2)) + [(s, s) for s in elements]

        # Calculate PRDF using matminer
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
        st.error(f"Error in PRDF calculation: {e}")
        return None, None, None


def calculate_and_display_sqs_prdf(sqs_structure, cutoff=10.0, bin_size=0.1):
    try:
        with st.expander("📊 PRDF Analysis of Generated SQS", expanded=True):
            with st.spinner("Calculating PRDF..."):
                prdf_dict, distance_dict, species_combinations = calculate_sqs_prdf(
                    sqs_structure, cutoff=cutoff, bin_size=bin_size
                )

                if prdf_dict is not None:
                    import plotly.graph_objects as go
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

                    import base64

                    st.write("**Download PRDF Data:**")
                    download_cols = st.columns(min(len(prdf_dict), 4))  # Max 4 columns

                    for idx, (pair, prdf_values) in enumerate(prdf_dict.items()):
                        df = pd.DataFrame()
                        df["Distance (Å)"] = distance_dict[pair]
                        df["PRDF"] = prdf_values

                        csv = df.to_csv(index=False)
                        filename = f"SQS_{pair[0]}_{pair[1]}_prdf.csv"

                        with download_cols[idx % len(download_cols)]:
                            st.download_button(
                                label=f"📥 {pair[0]}-{pair[1]} PRDF",
                                data=csv,
                                file_name=filename,
                                mime="text/csv",
                                key=f"download_prdf_{pair[0]}_{pair[1]}"
                            )

                    return True

                else:
                    st.error("Failed to calculate PRDF")
                    return False

    except Exception as e:
        st.error(f"Error calculating PRDF: {e}")
        return False


def render_atat_sqs_section():
    if 'full_structures' not in st.session_state or not st.session_state['full_structures']:
        st.warning("Please upload at least one structure file to use the ATAT SQS tool.")
        return

    file_options = list(st.session_state['full_structures'].keys())
    selected_atat_file = st.selectbox(
        "Select structure for ATAT SQS input generation:",
        file_options,
        key="atat_structure_selector"
    )

    if not selected_atat_file:
        return

    atat_structure = st.session_state['full_structures'][selected_atat_file]

    st.write(f"**Selected structure:** {atat_structure.composition.reduced_formula}")
    st.write(f"**Number of atoms:** {len(atat_structure)}")

    # tabs2, tabs4, tabs1, tabs5 = st.tabs([
    #    "1️⃣ + 2️⃣ + 3️⃣ Composition & Supercell ",
    #    "4️⃣ Clusters & Generation",
    #    "📊 Initial Structure View",
    #    "📊 Analyze ATAT Outputs"
    # ])

    col1, col2 = st.columns([1, 1])
    with col1:
        reduce_to_primitive = st.checkbox(
            "Convert to primitive cell before ATAT input generation",
            value=False,
            help="This will convert the structure to its primitive cell before creating ATAT input.",
            key="atat_reduce_primitive"
        )

        if reduce_to_primitive:
            analyzer = SpacegroupAnalyzer(atat_structure)
            primitive_structure = analyzer.get_primitive_standard_structure()
            st.write(f"**Primitive cell contains {len(primitive_structure)} atoms**")
            working_structure = primitive_structure
        else:
            working_structure = atat_structure

        try:
            analyzer = SpacegroupAnalyzer(working_structure)
            spg_symbol = analyzer.get_space_group_symbol()
            spg_number = analyzer.get_space_group_number()
            st.write(f"**Space group:** {spg_symbol} (#{spg_number})")
        except:
            st.write("**Space group:** Could not determine")

        unique_sites = get_unique_sites(working_structure)
        all_sites = get_all_sites(working_structure)

        st.subheader("Wyckoff Positions Analysis")

        site_data = []
        for site_info in unique_sites:
            site_data.append({
                "Wyckoff Index": site_info['wyckoff_index'],
                "Wyckoff Letter": site_info['wyckoff_letter'],
                "Current Element": site_info['element'],
                "Coordinates": f"({site_info['coords'][0]:.3f}, {site_info['coords'][1]:.3f}, {site_info['coords'][2]:.3f})",
                "Multiplicity": site_info['multiplicity'],
                "Site Indices": str(site_info['equivalent_indices'])
            })

        site_df = pd.DataFrame(site_data)
        st.dataframe(site_df, use_container_width=True)

    with col2:
        structure_preview(working_structure)

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #8B0000; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("🔵1️⃣ Step 1: Select Composition Mode")
    colb1, colb2 = st.columns([1, 1])
    with colb1:
        composition_mode = st.radio(
            "Choose composition specification mode:",
            [
                "🔄 Global Composition",
                "🎯 Sublattice-Specific"
            ],
            index=1,
            key="atat_composition_mode_radio",
            help="Global: Specify overall composition. Sublattice: Control each atomic position separately."
        )
    with colb2:
        with st.expander("ℹ️ Composition Mode Details", expanded=False):
            st.markdown("""
            ##### 🔄 Global Composition
            - Specify the target composition for the entire structure (e.g., 50% Fe, 50% Ni)
            - All crystallographic sites can be occupied by any of the selected elements
            - Elements are distributed randomly throughout the structure according to the specified fractions
            - **Example:** Fe₀.₅Ni₀.₅ random alloy where Fe and Ni atoms can occupy any position

            ---

            ##### 🎯 Sublattice-Specific  
            - Control which elements can occupy specific crystallographic sites (Wyckoff positions)
            - Set different compositions for different atomic sublattices
            - **Example:** In a perovskite ABO₃, control A-site (Ba/Sr) and B-site (Ti/Zr) compositions independently

            ---
            **Global** treats all sites equally, while **Sublattice-Specific** allows site-dependent element distributions.
            """)

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #8B0000; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("🔵2️⃣ Step 2: Supercell Configuration")

    col_x, col_y, col_z = st.columns(3)
    with col_x:
        nx = st.number_input("x-axis multiplier", value=2, min_value=1, max_value=10, step=1, key="atat_nx")
    with col_y:
        ny = st.number_input("y-axis multiplier", value=2, min_value=1, max_value=10, step=1, key="atat_ny")
    with col_z:
        nz = st.number_input("z-axis multiplier", value=2, min_value=1, max_value=10, step=1, key="atat_nz")

    transformation_matrix = np.array([
        [nx, 0, 0],
        [0, ny, 0],
        [0, 0, nz]
    ])

    st.write(f"**Supercell size:** {nx}×{ny}×{nz}")

    ase_atoms = pymatgen_to_ase(working_structure)
    supercell_preview = make_supercell(ase_atoms, transformation_matrix)
    st.write(f"**Preview: Supercell will contain {len(supercell_preview)} atoms**")

    all_elements = set()
    for site in working_structure:
        if site.is_ordered:
            all_elements.add(site.specie.symbol)
        else:
            for sp in site.species:
                all_elements.add(sp.symbol)

    use_sublattice_mode = composition_mode.startswith("🎯")
    target_concentrations = {}
    chem_symbols = None
    otrs = None

    supercell_multiplicity = nx * ny * nz
    total_supercell_atoms = len(supercell_preview)

    if composition_mode == "🔄 Global Composition":
        css = '''
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.15rem !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #f0f4ff !important;
            border-radius: 12px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
            border: none !important;
            color: #1e3a8a !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #dbe5ff !important;
            cursor: pointer;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #e0e7ff !important;
            color: #1e3a8a !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
        
            /* Added underline (thicker) */
            border-bottom: 4px solid #1e3a8a !important;
            border-radius: 12px 12px 0 0 !important; /* keep rounded only on top */
        }
        
        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        </style>
        '''

        st.markdown(css, unsafe_allow_html=True)
        common_elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Vac'
        ]

        all_elements_list = sorted(common_elements)

        structure_elements = set()
        for site in working_structure:
            if site.is_ordered:
                structure_elements.add(site.specie.symbol)
            else:
                for sp in site.species:
                    structure_elements.add(sp.symbol)

        st.markdown(
            """
            <hr style="border: none; height: 6px; background-color: #8B0000; border-radius: 8px; margin: 20px 0;">
            """,
            unsafe_allow_html=True
        )

        st.subheader("🔵3️⃣ Step 3: Select Elements and Concentrations")
        element_list = st.multiselect(
            "Select elements for ATAT SQS",
            options=all_elements_list,
            default=sorted(list(structure_elements)),
            key="atat_composition_global",
            help="Example: Select 'Fe' and 'Ni' for Fe-Ni alloy"
        )

        if len(element_list) == 0:
            st.error("You must select at least one element.")
            st.stop()

        composition_input = ", ".join(element_list)

        st.info(f"""
        **Global Mode Concentration Constraints:**
        - Supercell multiplicity: {supercell_multiplicity} (={nx}×{ny}×{nz})
        - Valid concentrations must be multiples of 1/{supercell_multiplicity}
        - Minimum step: 1/{supercell_multiplicity} = {1 / supercell_multiplicity:.6f}
        - Each concentration applies to ALL atomic sites equally
        """)

        st.write("**Set target composition fractions:**")
        cols = st.columns(len(element_list))
        target_concentrations = {}

        remaining = 1.0
        for j, elem in enumerate(element_list[:-1]):
            with cols[j]:
                min_step = 1.0 / supercell_multiplicity
                frac_val = st.slider(
                    f"{elem}:",
                    min_value=0.0,
                    max_value=remaining,
                    value=min(int(supercell_multiplicity / len(element_list)) * min_step, remaining),
                    step=min_step,
                    format="%.6f",
                    key=f"atat_comp_global_{elem}"
                )
                target_concentrations[elem] = frac_val
                remaining -= frac_val

        if element_list:
            last_elem = element_list[-1]
            target_concentrations[last_elem] = max(0.0, remaining)
            with cols[-1]:
                st.write(f"**{last_elem}: {target_concentrations[last_elem]:.6f}**")

        corrected_concentrations = {}
        corrections_made = False

        for elem, frac in target_concentrations.items():
            nearest_step = round(frac * supercell_multiplicity) / supercell_multiplicity
            corrected_concentrations[elem] = nearest_step
            if abs(frac - nearest_step) > 1e-6:
                corrections_made = True
                st.warning(
                    f"⚠️ {elem} concentration adjusted from {frac:.6f} to {nearest_step:.6f} (nearest valid value)")

        total_corrected = sum(corrected_concentrations.values())
        if abs(total_corrected - 1.0) > 1e-6:
            largest_elem = max(corrected_concentrations.keys(), key=lambda x: corrected_concentrations[x])
            adjustment = 1.0 - total_corrected
            corrected_concentrations[largest_elem] += adjustment
            if corrections_made:
                st.info(
                    f"Final adjustment: {largest_elem} = {corrected_concentrations[largest_elem]:.6f} to ensure total = 1.0")

        target_concentrations = corrected_concentrations

        if corrections_made:
            st.success("✅ All concentrations are now valid multiples of 1/{} = {:.6f}".format(
                supercell_multiplicity, 1 / supercell_multiplicity))

    else:
        element_list = [2, 2]
        composition_input = []
        chem_symbols, target_concentrations, otrs = render_site_sublattice_selector_fixed(
            working_structure, all_sites, unique_sites, supercell_multiplicity
        )

    if composition_mode == "🔄 Global Composition":

        try:
            achievable_concentrations_global, achievable_counts_global = calculate_achievable_concentrations(
                target_concentrations, supercell_multiplicity)

            st.write("**Overall Target vs. Achievable Concentrations:**")
            conc_data = []
            for element, target_frac in target_concentrations.items():
                achievable_frac = achievable_concentrations_global.get(element, 0)
                achievable_count = achievable_counts_global.get(element, 0)
                status = "✅ Exact" if abs(target_frac - achievable_frac) < 1e-6 else "⚠️ Rounded"

                total_element_atoms = achievable_count * len(working_structure)

                conc_data.append({
                    "Element": element,
                    "Target (%)": f"{target_frac * 100:.3f}",
                    "Achievable (%)": f"{achievable_frac * 100:.3f}",
                    # "Atoms per Site": achievable_count,
                    "Total Atoms": total_element_atoms,
                    # "Status": status
                })
            conc_df = pd.DataFrame(conc_data)
            st.dataframe(conc_df, use_container_width=True)
            st.write("**Per-Site Concentrations (All sites identical in Global Mode):**")

            preview_data = []
            for site_info in unique_sites:
                site_label = f"{site_info['element']} @ {site_info['wyckoff_letter']} (×{site_info['multiplicity']})"

                conc_parts = []
                for element, frac in sorted(achievable_concentrations_global.items()):
                    if frac > 1e-6:
                        conc_parts.append(f"{element}={frac:.6f}")

                preview_data.append({
                    "Wyckoff Position": site_label,
                    "Supercell Replicas": f"{supercell_multiplicity}",
                    "Site Concentrations": ", ".join(conc_parts),
                    "Note": "Same for all sites"
                })

            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)
            st.info("In Global Mode, all atomic sites receive identical concentration assignments.")
            st.write("#### **Overall Expected Element Distribution in Supercell:**")

            total_element_counts = {}
            for elem, per_site_count in achievable_counts_global.items():
                total_element_counts[elem] = per_site_count * len(working_structure)

            if total_element_counts:
                cols = st.columns(min(len(total_element_counts), 4))
                for i, (elem, count) in enumerate(sorted(total_element_counts.items())):
                    percentage = (count / total_supercell_atoms) * 100 if total_supercell_atoms > 0 else 0
                    with cols[i % len(cols)]:
                        if percentage >= 80:
                            color = "#2E4057"  # Dark Blue-Gray
                        elif percentage >= 60:
                            color = "#4A6741"  # Dark Forest Green
                        elif percentage >= 40:
                            color = "#6B73FF"  # Purple-Blue
                        elif percentage >= 25:
                            color = "#FF8C00"  # Dark Orange
                        elif percentage >= 15:
                            color = "#4ECDC4"  # Teal
                        elif percentage >= 10:
                            color = "#45B7D1"  # Blue
                        elif percentage >= 5:
                            color = "#96CEB4"  # Green
                        elif percentage >= 2:
                            color = "#FECA57"  # Yellow
                        elif percentage >= 1:
                            color = "#DDA0DD"  # Plum
                        else:
                            color = "#D3D3D3"  # Light Gray

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {color}, {color}CC);
                            padding: 20px;
                            border-radius: 15px;
                            text-align: center;
                            margin: 10px 0;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                            border: 2px solid rgba(255,255,255,0.2);
                        ">
                            <h1 style="
                                color: white;
                                font-size: 3em;
                                margin: 0;
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                font-weight: bold;
                            ">{elem}</h1>
                            <h2 style="
                                color: white;
                                font-size: 2em;
                                margin: 10px 0 0 0;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                            ">{percentage:.1f}%</h2>
                            <p style="
                                color: white;
                                font-size: 1.8em;
                                margin: 5px 0 0 0;
                                opacity: 0.9;
                            ">{int(round(count, 0))} atoms</p>
                        </div>
                        """, unsafe_allow_html=True)
                st.write(f"**Total expected atoms in supercell:** {int(total_supercell_atoms)}")

        except Exception as e:
            st.error(f"Error creating concentration preview: {e}")
    else:
        display_sublattice_preview_fixed(target_concentrations, chem_symbols, transformation_matrix, working_structure,
                                         unique_sites)

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #8B0000; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    st.subheader("🔵4️⃣ Step 4: ATAT Cluster Configuration")

    col_nn_btn, col_nn_results = st.columns([1, 3])

    with col_nn_btn:
        if st.button("🔍 Calculate NN Distances", type="secondary", key="calc_nn_atat"):
            with st.spinner("Calculating..."):
                nn_results = calculate_first_six_nn_atat_aware(
                    working_structure,
                    chem_symbols if use_sublattice_mode else None,
                    use_sublattice_mode,
                )

            st.session_state['nn_results'] = nn_results

    with col_nn_results:
        if 'nn_results' in st.session_state and st.session_state['nn_results']:
            nn_data = st.session_state['nn_results']

            if 'message' in nn_data:
                st.warning(nn_data['message'])
            else:
                if 'active_sites' in nn_data:
                    active_count = len(nn_data['active_sites'])
                    total_count = nn_data['total_sites']
                    active_site_names = []
                    if use_sublattice_mode and chem_symbols:
                        for i in nn_data['active_sites']:
                            if i < len(chem_symbols):
                                elements = "+".join(sorted(chem_symbols[i]))
                                active_site_names.append(elements)

                    st.info(
                        f"**Active sites:** {active_count}/{total_count} ({', '.join(set(active_site_names))} positions)")

                if nn_data['overall']:
                    st.write(
                        "**NN Distances Between Active Sites (unit cell normalized to the maximum lattice parameter):**")
                    overall_text = []
                    ordinals = {1: 'st', 2: 'nd', 3: 'rd', 4: 'th', 5: 'th', 6: 'th'}
                    for shell in nn_data['overall']:
                        ordinal = ordinals.get(shell['shell'], 'th')
                        overall_text.append(f"**{shell['shell']}{ordinal} NN:** {shell['distance']:.4f}")
                    st.write(" | ".join(overall_text))

                st.caption("💡 These values can suggest how to set the pair/triplet cut-off distances")

    col_cut1, col_cut2, col_cut3 = st.columns(3)
    with col_cut1:
        pair_cutoff = st.number_input(
            "Pair cutoff distance:",
            min_value=0.1,
            max_value=5.0,
            value=1.5,
            step=0.1,
            format="%.1f",
            help="Maximum distance for pair correlations. Usually 1.1 includes first 2 nearest neighbor shells.",
            key="atat_pair_cutoff"
        )

    with col_cut2:
        include_triplets = st.checkbox("Include triplet clusters", value=False, key="atat_include_triplets")
        if include_triplets:
            triplet_cutoff = st.number_input(
                "Triplet cutoff:",
                min_value=0.1,
                max_value=3.0,
                value=1.2,
                step=0.1,
                format="%.1f",
                key="atat_triplet_cutoff_val"
            )
        else:
            triplet_cutoff = None

    with col_cut3:
        include_quadruplets = st.checkbox("Include quadruplet clusters", value=False, key="atat_include_quadruplets")
        if include_quadruplets:
            quadruplet_cutoff = st.number_input(
                "Quadruplet cutoff:",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                format="%.1f",
                key="atat_quadruplet_cutoff_val"
            )
        else:
            quadruplet_cutoff = None

    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #ff6600; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    if "atat_results" not in st.session_state:
        st.session_state.atat_results = None

    current_config_key = f"{selected_atat_file}_{reduce_to_primitive}_{nx}_{ny}_{nz}_{str(target_concentrations)}_{composition_mode}_{pair_cutoff}_{triplet_cutoff}_{quadruplet_cutoff}"

    if "atat_config_key" not in st.session_state:
        st.session_state.atat_config_key = current_config_key
    elif st.session_state.atat_config_key != current_config_key:
        st.session_state.atat_results = None
        st.session_state.atat_config_key = current_config_key

    col_button, col_clear = st.columns([3, 1])

    with col_button:
        if not target_concentrations:
            st.warning("Create at least 1 sublattice (with minimum of two elements) first.")
            generate_atat_button = st.button("🔧 Generate ATAT Input Files", type="tertiary", disabled=True,
                                             help="Configure at least 1 sublattice concentration first.")
        elif len(element_list) < 2 and composition_mode == "🔄 Global Composition":
            st.warning(f"Select at least two elements first in Step 4:")
            generate_atat_button = st.button("🔧 Generate ATAT Input Files", type="tertiary", disabled=True,
                                             help="Select at least two elements first.")
        else:
            generate_atat_button = st.button("🔧 Generate ATAT Input Files", type="tertiary")

    with col_clear:
        if st.session_state.atat_results is not None:
            if st.button("🗑️ Clear Results", type="secondary", help="Clear current ATAT results"):
                st.session_state.atat_results = None
                st.rerun()

    if generate_atat_button:
        try:
            if composition_mode == "🔄 Global Composition":
                achievable_concentrations_for_atat, achievable_counts = calculate_achievable_concentrations(
                    target_concentrations, supercell_multiplicity)

                use_concentrations = achievable_concentrations_for_atat
                use_sublattice_mode_final = False
                use_chem_symbols = None
            else:
                achievable_concentrations_for_atat, adjustment_info = calculate_achievable_concentrations_sublattice_fixed(
                    target_concentrations, chem_symbols, transformation_matrix, working_structure, unique_sites
                )

                use_concentrations = achievable_concentrations_for_atat
                use_sublattice_mode_final = True
                use_chem_symbols = chem_symbols

            rndstr_content, sqscell_content, atat_commands, final_concentrations, adjustment_info = generate_atat_input_files_corrected(
                working_structure,
                use_concentrations,
                transformation_matrix,
                use_sublattice_mode_final,
                use_chem_symbols,
                nx, ny, nz,
                pair_cutoff,
                triplet_cutoff,
                quadruplet_cutoff,
                len(supercell_preview))

            if adjustment_info and len(adjustment_info) > 0:
                st.warning(
                    "⚠️ **Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
                adj_df = pd.DataFrame(adjustment_info)
                st.dataframe(adj_df, use_container_width=True)

            st.session_state.atat_results = {
                'structure_name': selected_atat_file,
                'supercell_size': f"{nx}×{ny}×{nz}",
                'total_atoms': len(supercell_preview),
                'pair_cutoff': pair_cutoff,
                'triplet_cutoff': triplet_cutoff,
                'quadruplet_cutoff': quadruplet_cutoff,
                'rndstr_content': rndstr_content,
                'sqscell_content': sqscell_content,
                'atat_commands': atat_commands,
                'final_concentrations': final_concentrations,
                'max_param': max(working_structure.lattice.a, working_structure.lattice.b, working_structure.lattice.c)
            }

            st.success("✅ ATAT input files generated successfully with corrected per-site concentrations!")
            st.rerun()

        except Exception as e:
            st.error(f"Error generating ATAT input files: {str(e)}")
            st.exception(e)
    if st.session_state.atat_results is not None:
        results = st.session_state.atat_results

        st.markdown("---")
        st.subheader("📊 Generated ATAT Configuration")

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Structure", results['structure_name'].split('.')[0])
            st.metric("Total Atoms", results['total_atoms'])
        with col_info2:
            st.metric("Supercell Size", results['supercell_size'])
            st.metric("Pair Cutoff", f"{results['pair_cutoff']:.1f}")
        with col_info3:
            if results['triplet_cutoff']:
                st.metric("Triplet Cutoff", f"{results['triplet_cutoff']:.1f}")
            if results['quadruplet_cutoff']:
                st.metric("Quadruplet Cutoff", f"{results['quadruplet_cutoff']:.1f}")

        cutoffs = [results['pair_cutoff']]
        if results.get('triplet_cutoff'):
            cutoffs.append(results['triplet_cutoff'])
        if results.get('quadruplet_cutoff'):
            cutoffs.append(results['quadruplet_cutoff'])

        render_concentration_sweep_section(
            chem_symbols,
            target_concentrations,
            transformation_matrix,
            working_structure,
            cutoffs, len(supercell_preview)
        )

        st.subheader("📁 Generated Files")
        col_file1, col_file2 = st.columns(2)

        with col_file1:
            st.write("**📄 rndstr.in**")
            st.code(results['rndstr_content'], language="text")
            st.download_button(
                label="📥 Download rndstr.in",
                data=results['rndstr_content'],
                file_name="rndstr.in",
                mime="text/plain",
                key="atat_download_rndstr_persistent",
                type="primary"
            )

        with col_file2:
            st.write("**📄 sqscell.out**")
            st.code(results['sqscell_content'], language="text")
            st.download_button(
                label="📥 Download sqscell.out",
                data=results['sqscell_content'],
                file_name="sqscell.out",
                mime="text/plain",
                key="atat_download_sqscell_persistent",
                type="primary"
            )

        st.subheader("🖥️ ATAT Commands to Run")
        st.code(results['atat_commands'], language="bash")

        with st.expander("📖 How to use these files with ATAT", expanded=False):

            st.markdown(f"""
            ### Steps to generate SQS with ATAT:

            1. **Download the files** above and place them in your ATAT working directory:
               - `rndstr.in` (structure definition with concentrations)
               - `sqscell.out` (supercell dimensions)

            2. **Generate clusters** using corrdump:
               ```bash
               corrdump -l=rndstr.in -ro -noe -nop -clus -2={results['pair_cutoff']}{' -3=' + str(results['triplet_cutoff']) if results['triplet_cutoff'] else ''}{' -4=' + str(results['quadruplet_cutoff']) if results['quadruplet_cutoff'] else ''}
               ```

            3. **Optional: View clusters**:
               ```bash
               getclus
               ```

            4. **Generate SQS** using mcsqs:
               ```bash
               mcsqs -rc
               ```
               OR specify atom count directly (will find the most randomized supercell that can accomodate {results['total_atoms']} atoms - distorts the original cell shape):
               ```bash
               mcsqs -n {results['total_atoms']}
               ```

            5. **Monitor progress**:
               - Watch `bestcorr.out` for correlation functions
               - Check `mcsqs.log` for objective function progress
               - Stop when correlation functions are acceptable (Ctrl+C)

            ### Expected Output Files:
            - **bestsqs.out** - Best SQS structure found
            - **bestcorr.out** - Correlation functions (monitor this!)
            - **mcsqs.log** - Progress log

            ### Tips:
            - **Binary alloys**: Usually converge in seconds to minutes
            - **Ternary alloys**: May take minutes to hours  
            - **Quaternary+ alloys**: Can take hours to days
            - **Parallel execution**: Use `-ip=1`, `-ip=2`, etc. for multiple instances
            - **Good objective function**: More negative is better (e.g., -0.95 > -0.85)
            - **Perfect match**: When all correlation differences in `bestcorr.out` are near zero

            ### Example parallel execution:
            ```bash
            mcsqs -rc -ip=1 &
            mcsqs -rc -ip=2 &  
            mcsqs -rc -ip=3 &
            wait
            ```

            ### Configuration Summary:
            - **Structure**: {results['structure_name']}
            - **Supercell**: {results['supercell_size']} ({results['total_atoms']} atoms)
            - **Pair cutoff**: {results['pair_cutoff']:.1f}
            {f"- **Triplet cutoff**: {results['triplet_cutoff']:.1f}" if results['triplet_cutoff'] else ""}
            {f"- **Quadruplet cutoff**: {results['quadruplet_cutoff']:.1f}" if results['quadruplet_cutoff'] else ""}
            """)

        render_monitor_script_section(results)

        st.markdown("---")
        col_status1, col_status2, col_status3 = st.columns(3)
        with col_status1:
            st.success("✅ Files Generated")
        with col_status2:
            st.info("📥 Ready for Download")
        with col_status3:
            st.info("🖥️ Commands Available")

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🔄 Analyze ATAT Outputs (convert bestsqs to VASP, LMP, CIF, XYZ, calculate PRDF, monitor logs)")
    st.info("Upload your ATAT output files to convert and analyze the results.")

    file_tab1, file_tab2 = st.tabs(
        ["📁 Structure Converter", "📊 Optimization Analysis (mcsqs.log, mcsqs_progress.csv, parallel runs...)"])

    with file_tab1:

        converter_mode = st.radio(
            "Choose conversion mode:",
            ["Single File Converter", "Batch Converter (Multiple Files)"],
            key="converter_mode_selector"
        )

        if converter_mode == "Single File Converter":
            st.write("**Upload bestsqs.out file to convert the output format:**")
            uploaded_bestsqs = st.file_uploader(
                "Upload bestsqs.out file:",
                type=['out', 'txt', 'log'],
                help="Upload the bestsqs.out file generated by ATAT mcsqs command",
                key="bestsqs_uploader"
            )

            if uploaded_bestsqs is not None:
                try:
                    bestsqs_content = uploaded_bestsqs.read().decode('utf-8')

                    if 'atat_results' in st.session_state and st.session_state.atat_results is not None:
                        results = st.session_state.atat_results
                    else:
                        results = {
                            'structure_name': selected_atat_file,
                            'supercell_size': f"{nx}×{ny}×{nz}" if 'nx' in locals() else "Unknown",
                            'total_atoms': len(supercell_preview) if 'supercell_preview' in locals() else 0
                        }

                    is_valid, validation_message = validate_bestsqs_file(bestsqs_content)

                    if not is_valid:
                        st.error(f"Invalid bestsqs.out file: {validation_message}")
                        st.info("Please ensure you upload a valid ATAT bestsqs.out file.")
                        # return

                    st.success(f"✅ Valid ATAT file detected: {validation_message}")
                    vasp_content, conversion_info = convert_bestsqs_to_vasp(
                        bestsqs_content,
                        working_structure,
                        transformation_matrix,
                        results['structure_name']
                    )

                    sqs_pymatgen_structure = convert_atat_to_pymatgen_structure(
                        bestsqs_content, working_structure, transformation_matrix
                    )

                    st.success("✅ Successfully converted bestsqs.out to VASP format!")
                    col_conv1, col_conv2 = st.columns(2)
                    with col_conv1:
                        st.write("#### **Conversion Summary:**")
                        for key, value in conversion_info.items():
                            st.write(f"- **{key}:** {value}")

                    with col_conv2:
                        st.write("#### **VASP POSCAR Preview:**")
                        preview_lines = vasp_content.split('\n')[:15]
                        st.code('\n'.join(preview_lines) + '\n...', language="text")
                    sqs_result = {
                        'structure': sqs_pymatgen_structure
                    }

                    st.write("#### **3D Structure Visualization:**")
                    sqs_visualization(sqs_result)

                    # Download buttons with multiple format options
                    # Download buttons with multiple format options
                    st.write("**Download Converted Structure:**")
                    col_down1, col_down2, col_down3 = st.columns(3)

                    with col_down1:
                        # VASP POSCAR download with options
                        st.markdown("**VASP Options:**")
                        use_fractional = st.checkbox("Output POSCAR with fractional coordinates",
                                                     value=True,
                                                     key="poscar_fractional")

                        from ase.constraints import FixAtoms
                        use_selective_dynamics = st.checkbox("Include Selective dynamics (all atoms free)",
                                                             value=False, key="poscar_sd")

                        try:
                            from pymatgen.core import Element

                            # Sort species by atomic weight
                            unique_species = []
                            for site in sqs_pymatgen_structure:
                                if site.specie not in unique_species:
                                    unique_species.append(site.specie)

                            species_weights = {}
                            for species in unique_species:
                                try:
                                    species_weights[species] = Element(species.symbol).atomic_mass
                                except:
                                    species_weights[species] = 999.0

                            sorted_species = sorted(unique_species, key=lambda x: species_weights[x])


                            new_struct = Structure(sqs_pymatgen_structure.lattice, [], [])
                            for species in sorted_species:
                                for site in sqs_pymatgen_structure:
                                    if site.specie == species:
                                        new_struct.append(
                                            species=site.species,
                                            coords=site.frac_coords,
                                            coords_are_cartesian=False,
                                        )

                            out = StringIO()
                            current_ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

                            if use_selective_dynamics:
                                constraint = FixAtoms(indices=[])  # No atoms are fixed, so all will be T T T
                                current_ase_structure.set_constraint(constraint)

                            write(out, current_ase_structure, format="vasp", direct=use_fractional, sort=False)
                            vasp_content_with_options = out.getvalue()

                            st.download_button(
                                label="📥 Download POSCAR",
                                data=vasp_content_with_options,
                                file_name=f"POSCAR_SQS_{results['structure_name'].split('.')[0]}.vasp",
                                mime="text/plain",
                                type="primary",
                                key="download_converted_poscar"
                            )
                        except Exception as e:
                            st.error(f"Error generating VASP file: {str(e)}")

                    with col_down2:
                        # Additional format selector
                        additional_format = st.selectbox(
                            "Additional Format:",
                            ["CIF", "LAMMPS", "XYZ"],
                            key="additional_format_selector"
                        )

                        # Show LAMMPS options if LAMMPS is selected
                        if additional_format == "LAMMPS":
                            st.markdown("**LAMMPS Export Options**")
                            atom_style = st.selectbox("Select atom_style", ["atomic", "charge", "full"], index=0,
                                                      key="lammps_atom_style")
                            units = st.selectbox("Select units", ["metal", "real", "si"], index=0, key="lammps_units")
                            include_masses = st.checkbox("Include atomic masses", value=True, key="lammps_masses")
                            force_skew = st.checkbox("Force triclinic cell (skew)", value=False, key="lammps_skew")

                    with col_down3:
                        if st.button("📄 Generate & Download", key="generate_additional_format"):
                            try:
                                if additional_format == "CIF":
                                    from pymatgen.io.cif import CifWriter

                                    # Create structure for CIF
                                    grouped_data = sqs_pymatgen_structure.copy()
                                    new_struct = Structure(sqs_pymatgen_structure.lattice, [], [])

                                    for site in sqs_pymatgen_structure:
                                        species_dict = {}
                                        for element, occupancy in site.species.items():
                                            species_dict[element] = float(occupancy)

                                        new_struct.append(
                                            species=species_dict,
                                            coords=site.frac_coords,
                                            coords_are_cartesian=False,
                                        )

                                    file_content = CifWriter(new_struct, symprec=0.1,
                                                             write_site_properties=True).__str__()
                                    download_file_name = f"{results['structure_name'].split('.')[0]}.cif"
                                    mime_type = "chemical/x-cif"

                                elif additional_format == "LAMMPS":
                                    # Create structure for LAMMPS
                                    new_struct = Structure(sqs_pymatgen_structure.lattice, [], [])

                                    for site in sqs_pymatgen_structure:
                                        new_struct.append(
                                            species=site.species,
                                            coords=site.frac_coords,
                                            coords_are_cartesian=False,
                                        )

                                    current_ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
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
                                    download_file_name = f"{results['structure_name'].split('.')[0]}.lmp"
                                    mime_type = "text/plain"

                                elif additional_format == "XYZ":
                                    # Generate XYZ format (you'll need to implement this)
                                    additional_content, additional_filename = generate_additional_format(
                                        sqs_pymatgen_structure, additional_format, results['structure_name']
                                    )
                                    file_content = additional_content
                                    download_file_name = additional_filename
                                    mime_type = get_mime_type(additional_format)

                                st.download_button(
                                    label=f"📥 Download {additional_format}",
                                    data=file_content,
                                    file_name=download_file_name,
                                    mime=mime_type,
                                    type="primary",
                                    key=f"download_{additional_format.lower()}"
                                )
                                st.success(f"✅ {additional_format} file generated!")

                            except Exception as e:
                                st.error(f"Error generating {additional_format}: {str(e)}")

                    st.write("**Complete Package:**")
                    if 'atat_results' in st.session_state and st.session_state.atat_results is not None:
                        zip_buffer_complete = create_complete_atat_zip(
                            st.session_state.atat_results, vasp_content, bestsqs_content
                        )

                        st.download_button(
                            label="📦 Download Complete Package",
                            data=zip_buffer_complete,
                            file_name=f"ATAT_SQS_Complete_{st.session_state.atat_results['structure_name'].split('.')[0]}.zip",
                            mime="application/zip",
                            type="primary",
                            key="download_complete_package"
                        )
                    else:
                        st.warning(
                            "⚠️ Complete package not available. Please generate ATAT input files in Step 4 first.")
                        st.button(
                            "📦 Complete Package (Unavailable)",
                            disabled=True,
                            help="Generate ATAT input files first to enable complete package download"
                        )
                    lattice1, lattice2, atoms = parse_atat_bestsqs_format(bestsqs_content)

                    element_counts = {}
                    for _, _, _, element in atoms:
                        element_counts[element] = element_counts.get(element, 0) + 1

                    # st.write("**Element Distribution:**")
                    # element_df = pd.DataFrame([
                    #     {"Element": elem, "Count": count, "Percentage": f"{count / len(atoms) * 100:.1f}%"}
                    #     for elem, count in sorted(element_counts.items())
                    # ])
                    # st.dataframe(element_df, use_container_width=True)

                    st.write("#### **Element Distribution:**")
                    cols = st.columns(min(len(element_counts), 4))  # Max 4 columns
                    for i, (elem, count) in enumerate(sorted(element_counts.items())):
                        percentage = count / len(atoms) * 100
                        with cols[i % len(cols)]:
                            if percentage >= 80:
                                color = "#2E4057"  # Dark Blue-Gray for very high concentration
                            elif percentage >= 60:
                                color = "#4A6741"  # Dark Forest Green for high concentration
                            elif percentage >= 40:
                                color = "#6B73FF"  # Purple-Blue for medium-high concentration
                            elif percentage >= 25:
                                color = "#FF8C00"  # Dark Orange for medium concentration
                            elif percentage >= 15:
                                color = "#4ECDC4"  # Teal for medium-low concentration
                            elif percentage >= 10:
                                color = "#45B7D1"  # Blue for low-medium concentration
                            elif percentage >= 5:
                                color = "#96CEB4"  # Green for low concentration
                            elif percentage >= 2:
                                color = "#FECA57"  # Yellow for very low concentration
                            elif percentage >= 1:
                                color = "#DDA0DD"  # Plum for trace concentration
                            else:
                                color = "#D3D3D3"  # Light Gray for minimal concentration

                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, {color}, {color}CC);
                                padding: 20px; 
                                border-radius: 15px; 
                                text-align: center; 
                                margin: 10px 0;
                                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                                border: 2px solid rgba(255,255,255,0.2);
                            ">
                                <h1 style="
                                    color: white; 
                                    font-size: 3em; 
                                    margin: 0; 
                                    text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                    font-weight: bold;
                                ">{elem}</h1>
                                <h2 style="
                                    color: white; 
                                    font-size: 2em; 
                                    margin: 10px 0 0 0;
                                    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                                ">{percentage:.1f}%</h2>
                                <p style="
                                    color: white; 
                                    font-size: 1.8em; 
                                    margin: 5px 0 0 0;
                                    opacity: 0.9;
                                ">{count} atoms</p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.subheader("📊 PRDF Analysis of SQS Structure")

                    col_prdf1, col_prdf2, col_prdf3 = st.columns(3)
                    with col_prdf1:
                        prdf_cutoff = st.number_input(
                            "PRDF cutoff distance (Å):",
                            min_value=1.0,
                            max_value=20.0,
                            value=10.0,
                            step=0.5,
                            key="prdf_cutoff"
                        )
                    with col_prdf2:
                        prdf_bin_size = st.number_input(
                            "Bin size (Å):",
                            min_value=0.01,
                            max_value=1.0,
                            value=0.1,
                            step=0.01,
                            key="prdf_bin_size"
                        )
                    with col_prdf3:
                        calculate_prdf_btn = st.button(
                            "🔬 Calculate PRDF",
                            type="secondary",
                            key="calculate_prdf_btn"
                        )

                    if calculate_prdf_btn:
                        try:
                            prdf_structure = prepare_structure_for_prdf(sqs_pymatgen_structure)
                            calculate_and_display_sqs_prdf(prdf_structure, prdf_cutoff, prdf_bin_size)

                        except Exception as prdf_error:
                            st.error(f"Error calculating PRDF: {str(prdf_error)}")
                            st.info("PRDF calculation requires a valid structure with multiple element types.")
                            import traceback
                            st.error(f"Debug: {traceback.format_exc()}")
                    render_vacancy_creation_section(sqs_pymatgen_structure)

                except UnicodeDecodeError:
                    st.error("Error reading file. Please ensure the file is a text file with UTF-8 encoding.")
                except Exception as e:
                    st.error(f"Error processing bestsqs.out file: {str(e)}")
                    st.error("Please ensure the file is a valid ATAT bestsqs.out format.")
                    import traceback
                    st.error(f"Debug info: {traceback.format_exc()}")
        else:
            render_batch_structure_converter(
                working_structure, transformation_matrix)
    with file_tab2:
        render_extended_optimization_analysis_tab()


def prepare_structure_for_prdf(structure):
    from pymatgen.core import Structure

    new_species = []
    new_coords = []

    for site in structure:
        if site.is_ordered:
            new_species.append(site.specie)
        else:
            dominant_species = max(site.species.items(), key=lambda x: x[1])[0]
            new_species.append(dominant_species)

        new_coords.append(site.frac_coords)
    ordered_structure = Structure(structure.lattice, new_species, new_coords)
    return ordered_structure


def generate_additional_format(structure, file_format, structure_name):
    from io import StringIO
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write

    base_name = structure_name.split('.')[0]

    if file_format == "CIF":
        from pymatgen.io.cif import CifWriter

        ordered_structure = prepare_structure_for_prdf(structure)

        file_content = CifWriter(ordered_structure, symprec=0.1).__str__()
        filename = f"SQS_{base_name}.cif"

    elif file_format == "LAMMPS":
        ordered_structure = prepare_structure_for_prdf(structure)
        ase_structure = AseAtomsAdaptor.get_atoms(ordered_structure)

        out = StringIO()
        write(
            out,
            ase_structure,
            format="lammps-data",
            atom_style="atomic",
            units="metal",
            masses=True,
            force_skew=False
        )
        file_content = out.getvalue()
        filename = f"SQS_{base_name}.lmp"

    elif file_format == "XYZ":
        ordered_structure = prepare_structure_for_prdf(structure)

        lattice_vectors = ordered_structure.lattice.matrix
        cart_coords = []
        elements = []

        for site in ordered_structure:
            cart_coords.append(ordered_structure.lattice.get_cartesian_coords(site.frac_coords))
            elements.append(site.specie.symbol)

        xyz_lines = []
        xyz_lines.append(str(len(ordered_structure)))

        # Extended XYZ format with lattice information
        lattice_string = " ".join([f"{x:.6f}" for row in lattice_vectors for x in row])
        properties = "Properties=species:S:1:pos:R:3"
        comment_line = f'Lattice="{lattice_string}" {properties}'
        xyz_lines.append(comment_line)

        for element, coord in zip(elements, cart_coords):
            line = f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
            xyz_lines.append(line)

        file_content = "\n".join(xyz_lines)
        filename = f"SQS_{base_name}.xyz"

    else:
        raise ValueError(f"Unsupported format: {file_format}")

    return file_content, filename


def get_mime_type(file_format):
    mime_types = {
        "CIF": "chemical/x-cif",
        "LAMMPS": "text/plain",
        "XYZ": "chemical/x-xyz"
    }
    return mime_types.get(file_format, "text/plain")


def convert_atat_to_pymatgen_structure(bestsqs_content, original_structure, transformation_matrix):
    from pymatgen.core import Structure
    import numpy as np

    A_basis = original_structure.lattice.matrix

    _, B_transform, atoms_in_A_coords = parse_atat_bestsqs_format(bestsqs_content)
    B = np.array(B_transform)

    final_lattice_vectors = np.dot(B, A_basis)
    cartesian_coords = []
    species = []

    for x, y, z, element in atoms_in_A_coords:
        if element.lower() in ['vac', "'vac", 'vacancy', 'x']:
            continue

        atom_coord_in_A = np.array([x, y, z])
        cart_pos = np.dot(atom_coord_in_A, A_basis)
        cartesian_coords.append(cart_pos)
        species.append(element)

    if not species:
        raise ValueError("All atoms in the structure are vacancies - cannot create a valid structure")

    sqs_structure = Structure(
        lattice=final_lattice_vectors,
        species=species,
        coords=cartesian_coords,
        coords_are_cartesian=True
    )

    return sqs_structure


def convert_bestsqs_to_vasp(bestsqs_content, original_structure, transformation_matrix, structure_name):
    from pymatgen.core import Lattice, Element
    import numpy as np

    A_basis = original_structure.lattice.matrix

    _, B_transform, atoms_in_A_coords = parse_atat_bestsqs_format(bestsqs_content)
    B = np.array(B_transform)

    final_lattice_vectors = np.dot(B, A_basis)

    atom_data = []
    vacancy_count = 0

    for x, y, z, element in atoms_in_A_coords:
        if element.lower() in ['vac', "'vac", 'vacancy', 'x']:
            vacancy_count += 1
            continue

        atom_coord_in_A = np.array([x, y, z])
        cart_pos = np.dot(atom_coord_in_A, A_basis)
        atom_data.append({'element': element, 'cart_pos': cart_pos})

    if not atom_data:
        raise ValueError("All atoms in the structure are vacancies - cannot create a valid POSCAR")

    from pymatgen.core import Element

    unique_elements_set = set(atom['element'] for atom in atom_data)

    element_weights = {}
    for elem in unique_elements_set:
        try:
            element_weights[elem] = Element(elem).atomic_mass
        except:
            element_weights[elem] = 999.0

    unique_elements = sorted(list(unique_elements_set), key=lambda x: element_weights[x])

    sorted_atoms_cart = []
    element_counts = []
    for element in unique_elements:
        atoms_of_element = [atom['cart_pos'] for atom in atom_data if atom['element'] == element]
        sorted_atoms_cart.extend(atoms_of_element)
        element_counts.append(len(atoms_of_element))

    inv_final_lattice = np.linalg.inv(final_lattice_vectors)
    fractional_coords = [np.dot(pos, inv_final_lattice) for pos in sorted_atoms_cart]

    comment = f"SQS from {structure_name} via ATAT"
    if vacancy_count > 0:
        comment += f" ({vacancy_count} vacancies removed)"

    poscar_lines = [comment, "1.0"]
    for vec in final_lattice_vectors:
        poscar_lines.append(f"  {vec[0]:15.9f} {vec[1]:15.9f} {vec[2]:15.9f}")

    poscar_lines.append(" ".join(unique_elements))
    poscar_lines.append(" ".join(map(str, element_counts)))
    poscar_lines.append("Direct")

    for pos in fractional_coords:
        poscar_lines.append(f"  {pos[0]:15.9f} {pos[1]:15.9f} {pos[2]:15.9f}")

    vasp_content = "\n".join(poscar_lines)

    supercell_lattice = Lattice(final_lattice_vectors)
    conversion_info = {
        "Source Structure": structure_name,
        "Total Original Atoms": len(atoms_in_A_coords),
        "Atoms After Removing Vacancies": len(atom_data),
        "Vacancies Removed": vacancy_count,
        "Elements & Counts": ", ".join([f"{elem}: {count}" for elem, count in zip(unique_elements, element_counts)]),
        "SQS Lattice (a, b, c)": f"{supercell_lattice.a:.4f} Å, {supercell_lattice.b:.4f} Å, {supercell_lattice.c:.4f} Å",
        "SQS Angles (α, β, γ)": f"{supercell_lattice.alpha:.2f}°, {supercell_lattice.beta:.2f}°, {supercell_lattice.gamma:.2f}°",
        "Element Order": f"Sorted by atomic weight: {' < '.join([f'{elem}({element_weights[elem]:.2f})' for elem in unique_elements])}"
    }

    return vasp_content, conversion_info


def debug_atat_conversion_step_by_step(bestsqs_content, original_structure):
    import numpy as np
    from pymatgen.core import Lattice

    lattice1, lattice2, atoms = parse_atat_bestsqs_format(bestsqs_content)

    print("=== Step-by-Step ATAT Conversion Debug ===")
    print(f"A (unit cell vectors, lines 1-3):")
    A = np.array(lattice1)
    for i, row in enumerate(A):
        print(f"  A[{i}] = [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

    print(f"\nB (SQS lattice vectors in unit cell coords, lines 4-6):")
    B = np.array(lattice2)
    for i, row in enumerate(B):
        print(f"  B[{i}] = [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

    print(f"\nStep 1: Calculate lattice vectors = B × A")
    lattice_vectors = np.dot(B, A)
    for i, row in enumerate(lattice_vectors):
        print(f"  Lattice[{i}] = [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

    print(f"\nStep 2: Sample atomic coordinate conversion (C × A)")
    print("First 3 atoms:")
    for i, (x, y, z, element) in enumerate(atoms[:3]):
        cart_pos = np.dot([x, y, z], A)
        print(
            f"  Atom {i + 1} ({element}): [{x:.6f}, {y:.6f}, {z:.6f}] → [{cart_pos[0]:.6f}, {cart_pos[1]:.6f}, {cart_pos[2]:.6f}]")

    supercell_lattice = Lattice(lattice_vectors)
    print(f"\nResulting lattice parameters:")
    print(f"  a = {supercell_lattice.a:.6f} Å")
    print(f"  b = {supercell_lattice.b:.6f} Å")
    print(f"  c = {supercell_lattice.c:.6f} Å")
    print(f"  α = {supercell_lattice.alpha:.1f}°")
    print(f"  β = {supercell_lattice.beta:.1f}°")
    print(f"  γ = {supercell_lattice.gamma:.1f}°")

    if original_structure:
        orig_lattice = original_structure.lattice
        print(f"\nOriginal structure lattice parameters:")
        print(f"  a = {orig_lattice.a:.6f} Å")
        print(f"  b = {orig_lattice.b:.6f} Å")
        print(f"  c = {orig_lattice.c:.6f} Å")
        print(f"  α = {orig_lattice.alpha:.1f}°")
        print(f"  β = {orig_lattice.beta:.1f}°")
        print(f"  γ = {orig_lattice.gamma:.1f}°")

        print(f"\nExpected supercell ratios:")
        print(f"  a_ratio = {supercell_lattice.a / orig_lattice.a:.2f}")
        print(f"  b_ratio = {supercell_lattice.b / orig_lattice.b:.2f}")
        print(f"  c_ratio = {supercell_lattice.c / orig_lattice.c:.2f}")

    return supercell_lattice


def debug_atat_conversion_with_original(bestsqs_content, original_structure):
    import numpy as np
    from pymatgen.core import Lattice

    lattice1, lattice2, atoms = parse_atat_bestsqs_format(bestsqs_content)

    print("=== ATAT Conversion Debug (with original structure) ===")
    print(f"ATAT dArrVec1 (normalized basis vectors):")
    for i, row in enumerate(lattice1):
        print(f"  [{i}] {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}")

    print(f"\nATAT dArrVec2 (supercell transformation):")
    for i, row in enumerate(lattice2):
        print(f"  [{i}] {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}")

    orig_lattice = original_structure.lattice
    print(f"\nOriginal structure lattice matrix:")
    for i, row in enumerate(orig_lattice.matrix):
        print(f"  [{i}] {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}")

    print(f"\nOriginal lattice parameters:")
    print(f"  a = {orig_lattice.a:.6f} Å")
    print(f"  b = {orig_lattice.b:.6f} Å")
    print(f"  c = {orig_lattice.c:.6f} Å")
    print(f"  α = {orig_lattice.alpha:.1f}°")
    print(f"  β = {orig_lattice.beta:.1f}°")
    print(f"  γ = {orig_lattice.gamma:.1f}°")

    dArrVec1 = orig_lattice.matrix
    dArrVec2 = np.array(lattice2)
    dArrLatVec = np.dot(dArrVec2, dArrVec1)

    print(f"\nCalculated supercell lattice vectors (using original lattice):")
    for i, row in enumerate(dArrLatVec):
        print(f"  [{i}] {row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}")

    supercell_lattice = Lattice(dArrLatVec)
    print(f"\nResulting supercell lattice parameters:")
    print(f"  a = {supercell_lattice.a:.6f} Å  (expected: {3 * orig_lattice.a:.6f})")
    print(f"  b = {supercell_lattice.b:.6f} Å  (expected: {3 * orig_lattice.b:.6f})")
    print(f"  c = {supercell_lattice.c:.6f} Å  (expected: {2 * orig_lattice.c:.6f})")
    print(f"  α = {supercell_lattice.alpha:.1f}°  (expected: {orig_lattice.alpha:.1f}°)")
    print(f"  β = {supercell_lattice.beta:.1f}°   (expected: {orig_lattice.beta:.1f}°)")
    print(f"  γ = {supercell_lattice.gamma:.1f}°  (expected: {orig_lattice.gamma:.1f}°)")

    return supercell_lattice


def render_batch_structure_converter(working_structure, transformation_matrix):
    st.subheader("🔄 Batch Structure Converter (Multiple Parallel Runs)")
    st.info("Upload multiple bestsqs.out files from parallel ATAT runs to convert them all at once.")

    uploaded_batch_files = st.file_uploader(
        "Upload multiple bestsqs.out files:",
        type=['out', 'txt', 'log'],
        accept_multiple_files=True,
        help="Upload multiple bestsqs.out files from parallel ATAT runs",
        key="batch_bestsqs_uploader"
    )

    if uploaded_batch_files and len(uploaded_batch_files) > 0:
        st.success(
            f"✅ {len(uploaded_batch_files)} files uploaded successfully!")

        valid_files = []
        for uploaded_file in uploaded_batch_files:
            try:
                file_content = uploaded_file.read().decode('utf-8')
                is_valid, validation_message = validate_bestsqs_file(
                    file_content)

                if is_valid:
                    valid_files.append({
                        'name': uploaded_file.name,
                        'content': file_content
                    })
                else:
                    st.warning(
                        f"Skipping {uploaded_file.name}: {validation_message}")
            except Exception as e:
                st.warning(f"Error reading {uploaded_file.name}: {str(e)}")

        if not valid_files:
            st.error("No valid bestsqs.out files found.")
            return

        st.success(
            f"✅ {len(valid_files)} valid ATAT files ready for conversion")

        file_preview_data = []
        for file_data in valid_files:
            lattice1, lattice2, atoms = parse_atat_bestsqs_format(
                file_data['content'])
            element_counts = {}
            for _, _, _, element in atoms:
                element_counts[element] = element_counts.get(element, 0) + 1

            composition_str = ", ".join(
                [f"{elem}: {count}" for elem, count in sorted(element_counts.items())])

            file_preview_data.append({
                "File": file_data['name'],
                "Total Atoms": len(atoms),
                "Composition": composition_str
            })

        preview_df = pd.DataFrame(file_preview_data)
        st.dataframe(preview_df, use_container_width=True)

        st.subheader("📋 Output Format Configuration")

        col_format1, col_format2, col_format3, col_format4 = st.columns(4)

        with col_format1:
            st.markdown("**VASP POSCAR Options:**")
            include_vasp = st.checkbox(
                "Include VASP POSCAR", value=True, key="batch_include_vasp")
            if include_vasp:
                vasp_fractional = st.checkbox(
                    "Fractional coordinates", value=True, key="batch_vasp_fractional")
                vasp_selective = st.checkbox(
                    "Selective dynamics", value=False, key="batch_vasp_selective")

        with col_format2:
            st.markdown("**CIF Options:**")
            include_cif = st.checkbox(
                "Include CIF", value=False, key="batch_include_cif")
            if include_cif:
                cif_symprec = st.number_input("Symmetry precision", value=0.1, min_value=0.001, max_value=1.0,
                                              step=0.001, format="%.3f", key="batch_cif_symprec")

        with col_format3:
            st.markdown("**LAMMPS Options:**")
            include_lammps = st.checkbox(
                "Include LAMMPS", value=False, key="batch_include_lammps")
            if include_lammps:
                lammps_atom_style = st.selectbox(
                    "Atom style", ["atomic", "charge", "full"], index=0, key="batch_lammps_style")
                lammps_units = st.selectbox(
                    "Units", ["metal", "real", "si"], index=0, key="batch_lammps_units")
                lammps_masses = st.checkbox(
                    "Include masses", value=True, key="batch_lammps_masses")
                lammps_skew = st.checkbox(
                    "Force triclinic", value=False, key="batch_lammps_skew")

        with col_format4:
            st.markdown("**XYZ Options:**")
            include_xyz = st.checkbox(
                "Include XYZ", value=False, key="batch_include_xyz")
            if include_xyz:
                xyz_extended = st.checkbox(
                    "Extended XYZ format", value=True, key="batch_xyz_extended")

        if not any([include_vasp, include_cif, include_lammps, include_xyz]):
            st.warning("Please select at least one output format.")
            return

        if st.button("🔄 Convert All Files", type="primary", key="batch_convert_all"):
            try:
                with st.spinner(f"Converting {len(valid_files)} files..."):
                    zip_buffer = create_batch_conversion_zip(
                        valid_files, working_structure, transformation_matrix,
                        include_vasp, include_cif, include_lammps, include_xyz,
                        vasp_fractional if include_vasp else None,
                        vasp_selective if include_vasp else None,
                        cif_symprec if include_cif else None,
                        lammps_atom_style if include_lammps else None,
                        lammps_units if include_lammps else None,
                        lammps_masses if include_lammps else None,
                        lammps_skew if include_lammps else None,
                        xyz_extended if include_xyz else None
                    )

                st.success("✅ All files converted successfully!")

                format_list = []
                if include_vasp:
                    format_list.append("VASP")
                if include_cif:
                    format_list.append("CIF")
                if include_lammps:
                    format_list.append("LAMMPS")
                if include_xyz:
                    format_list.append("XYZ")

                st.download_button(
                    label=f"📦 Download All ({', '.join(format_list)})",
                    data=zip_buffer,
                    file_name=f"batch_conversion_{len(valid_files)}_files.zip",
                    mime="application/zip",
                    type="primary",
                    key="download_batch_conversion"
                )

                st.info(
                    f"Package contains {len(valid_files)} structures in {len(format_list)} format(s)")

            except Exception as e:
                st.error(f"Error during batch conversion: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")


def create_batch_conversion_zip(valid_files, working_structure, transformation_matrix,
                                include_vasp, include_cif, include_lammps, include_xyz,
                                vasp_fractional, vasp_selective, cif_symprec,
                                lammps_atom_style, lammps_units, lammps_masses, lammps_skew,
                                xyz_extended):
    import zipfile
    from io import BytesIO, StringIO
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.cif import CifWriter
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write
    from ase.constraints import FixAtoms

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:

        summary_lines = ["BATCH CONVERSION SUMMARY", "=" * 40, ""]

        for i, file_data in enumerate(valid_files):
            file_name = file_data['name']
            file_content = file_data['content']
            base_name = file_name.replace('.out', '').replace('.txt', '')

            try:
                sqs_structure = convert_atat_to_pymatgen_structure(
                    file_content, working_structure, transformation_matrix
                )

                lattice1, lattice2, atoms = parse_atat_bestsqs_format(
                    file_content)
                element_counts = {}
                for _, _, _, element in atoms:
                    element_counts[element] = element_counts.get(
                        element, 0) + 1

                summary_lines.append(f"File {i + 1}: {file_name}")
                summary_lines.append(f"  Total atoms: {len(atoms)}")
                summary_lines.append(
                    f"  Composition: {', '.join([f'{elem}: {count}' for elem, count in sorted(element_counts.items())])}")
                summary_lines.append("")

                if include_vasp:
                    try:
                        from pymatgen.core import Element

                        # Sort species by atomic weight
                        unique_species = []
                        for site in sqs_structure:
                            if site.specie not in unique_species:
                                unique_species.append(site.specie)

                        species_weights = {}
                        for species in unique_species:
                            try:
                                species_weights[species] = Element(species.symbol).atomic_mass
                            except:
                                species_weights[species] = 999.0

                        sorted_species = sorted(unique_species, key=lambda x: species_weights[x])

                        new_struct = Structure(sqs_structure.lattice, [], [])
                        for species in sorted_species:
                            for site in sqs_structure:
                                if site.specie == species:
                                    new_struct.append(
                                        species=site.species,
                                        coords=site.frac_coords,
                                        coords_are_cartesian=False,
                                    )

                        ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

                        if vasp_selective:
                            constraint = FixAtoms(indices=[])
                            ase_structure.set_constraint(constraint)

                        out = StringIO()
                        write(out, ase_structure, format="vasp",
                              direct=vasp_fractional, sort=False)
                        vasp_content = out.getvalue()

                        zip_file.writestr(
                            f"VASP/{base_name}_POSCAR.vasp", vasp_content)
                    except Exception as e:
                        summary_lines.append(
                            f"  VASP conversion failed: {str(e)}")

                if include_cif:
                    try:
                        ordered_structure = prepare_structure_for_prdf(
                            sqs_structure)
                        new_struct = Structure(sqs_structure.lattice, [], [])

                        for site in sqs_structure:
                            species_dict = {}
                            for element, occupancy in site.species.items():
                                species_dict[element] = float(occupancy)

                            new_struct.append(
                                species=species_dict,
                                coords=site.frac_coords,
                                coords_are_cartesian=False,
                            )

                        cif_content = CifWriter(
                            new_struct, symprec=cif_symprec, write_site_properties=True).__str__()
                        zip_file.writestr(f"CIF/{base_name}.cif", cif_content)
                    except Exception as e:
                        summary_lines.append(
                            f"  CIF conversion failed: {str(e)}")

                if include_lammps:
                    try:
                        new_struct = Structure(sqs_structure.lattice, [], [])
                        for site in sqs_structure:
                            new_struct.append(
                                species=site.species,
                                coords=site.frac_coords,
                                coords_are_cartesian=False,
                            )

                        ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
                        out = StringIO()
                        write(
                            out, ase_structure, format="lammps-data",
                            atom_style=lammps_atom_style, units=lammps_units,
                            masses=lammps_masses, force_skew=lammps_skew
                        )
                        lammps_content = out.getvalue()
                        zip_file.writestr(
                            f"LAMMPS/{base_name}.lmp", lammps_content)
                    except Exception as e:
                        summary_lines.append(
                            f"  LAMMPS conversion failed: {str(e)}")

                if include_xyz:
                    try:
                        ordered_structure = prepare_structure_for_prdf(
                            sqs_structure)

                        lattice_vectors = ordered_structure.lattice.matrix
                        cart_coords = []
                        elements = []

                        for site in ordered_structure:
                            cart_coords.append(
                                ordered_structure.lattice.get_cartesian_coords(site.frac_coords))
                            elements.append(site.specie.symbol)

                        xyz_lines = []
                        xyz_lines.append(str(len(ordered_structure)))

                        if xyz_extended:
                            lattice_string = " ".join(
                                [f"{x:.6f}" for row in lattice_vectors for x in row])
                            properties = "Properties=species:S:1:pos:R:3"
                            comment_line = f'Lattice="{lattice_string}" {properties}'
                            xyz_lines.append(comment_line)
                        else:
                            xyz_lines.append(f"Generated from {file_name}")

                        for element, coord in zip(elements, cart_coords):
                            line = f"{element} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
                            xyz_lines.append(line)

                        xyz_content = "\n".join(xyz_lines)
                        zip_file.writestr(f"XYZ/{base_name}.xyz", xyz_content)
                    except Exception as e:
                        summary_lines.append(
                            f"  XYZ conversion failed: {str(e)}")

            except Exception as e:
                summary_lines.append(
                    f"  Structure conversion failed: {str(e)}")

        summary_lines.extend([
            "", "CONVERSION SETTINGS:",
            f"VASP: {'Enabled' if include_vasp else 'Disabled'}",
            f"CIF: {'Enabled' if include_cif else 'Disabled'}",
            f"LAMMPS: {'Enabled' if include_lammps else 'Disabled'}",
            f"XYZ: {'Enabled' if include_xyz else 'Disabled'}",
            "",
            f"Total files processed: {len(valid_files)}",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])

        zip_file.writestr("README.txt", "\n".join(summary_lines))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def create_complete_atat_zip(results, vasp_content, bestsqs_content):
    import zipfile
    from io import BytesIO

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Original ATAT input files
        zip_file.writestr("1_INPUT_FILES/rndstr.in", results['rndstr_content'])
        zip_file.writestr("1_INPUT_FILES/sqscell.out", results['sqscell_content'])
        zip_file.writestr("1_INPUT_FILES/atat_commands.sh", results['atat_commands'])

        # ATAT output file
        zip_file.writestr("2_ATAT_OUTPUT/bestsqs.out", bestsqs_content)

        # Converted VASP file
        zip_file.writestr("3_VASP_CONVERTED/POSCAR", vasp_content)

        readme_content = f"""
# ATAT SQS Complete Package

## Contents:
- `1_INPUT_FILES/`: Original ATAT input files
  - `rndstr.in`: Structure definition with concentrations
  - `sqscell.out`: Supercell transformation matrix
  - `atat_commands.sh`: Command sequence for ATAT

- `2_ATAT_OUTPUT/`: 
  - `bestsqs.out`: Best SQS structure from ATAT

- `3_VASP_CONVERTED/`:
  - `POSCAR`: Converted structure for VASP calculations

## Configuration Summary:
- Structure: {results['structure_name']}
- Supercell: {results['supercell_size']} ({results['total_atoms']} atoms)
- Pair cutoff: {results['pair_cutoff']:.1f} Å
{f"- Triplet cutoff: {results['triplet_cutoff']:.1f} Å" if results['triplet_cutoff'] else ""}
{f"- Quadruplet cutoff: {results['quadruplet_cutoff']:.1f} Å" if results['quadruplet_cutoff'] else ""}

## Usage:
1. Use files in `1_INPUT_FILES/` to run ATAT
2. Compare your output with `2_ATAT_OUTPUT/bestsqs.out`
3. Use `3_VASP_CONVERTED/POSCAR` for DFT calculations

Generated by ATAT SQS Input File Generator
"""
        zip_file.writestr("README.txt", readme_content)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def parse_atat_bestsqs_format(content):
    lines = content.strip().split('\n')

    lattice1 = []
    lattice2 = []

    for i in range(3):
        lattice1.append([float(x) for x in lines[i].split()])

    for i in range(3, 6):
        lattice2.append([float(x) for x in lines[i].split()])

    atoms = []
    for i in range(6, len(lines)):
        line = lines[i].strip()
        if line:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                element = parts[3]
                atoms.append((x, y, z, element))

    return lattice1, lattice2, atoms


def validate_bestsqs_file(content):
    try:
        lines = content.strip().split('\n')

        if len(lines) < 7:
            return False, "File too short - needs at least 6 lattice lines + atomic positions"

        for i in range(6):
            parts = lines[i].split()
            if len(parts) != 3:
                return False, f"Line {i + 1} should contain exactly 3 numbers (lattice vector)"
            try:
                [float(x) for x in parts]
            except ValueError:
                return False, f"Line {i + 1} contains non-numeric values"

        atom_count = 0
        vacancy_count = 0

        for i in range(6, len(lines)):
            line = lines[i].strip()
            if line:
                parts = line.split()
                if len(parts) < 4:
                    return False, f"Line {i + 1} should contain x y z element"
                try:
                    float(parts[0]), float(parts[1]), float(parts[2])
                    element = parts[3]
                    if element.lower() in ['vac', "'vac", 'vacancy', 'x']:
                        vacancy_count += 1
                    else:
                        atom_count += 1
                except ValueError:
                    return False, f"Line {i + 1} contains invalid coordinates"

        total_sites = atom_count + vacancy_count
        if total_sites == 0:
            return False, "No valid atomic positions found"

        message = f"Valid ATAT file with {atom_count} atoms"
        if vacancy_count > 0:
            message += f" and {vacancy_count} vacancies"

        return True, message

    except Exception as e:
        return False, f"Error parsing file: {str(e)}"


def parse_mcsqs_log(log_content):
    import re

    lines = log_content.strip().split('\n')
    objective_values = []
    correlation_data = []

    for line in lines:
        line = line.strip()
        if "Objective_function=" in line:
            match = re.search(r'Objective_function=\s*([-+]?\d*\.?\d+)', line)
            if match:
                objective_values.append(float(match.group(1)))

        if "Correlations_mismatch=" in line:
            match = re.search(r'Correlations_mismatch=\s*(.*?)\s*>', line)
            if match:
                correlation_str = match.group(1)
                try:
                    correlations = [float(x) for x in correlation_str.split() if x.strip()]
                    correlation_data.append(correlations)
                except ValueError:
                    continue

    return objective_values, correlation_data


def create_optimization_plot(objective_values):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    steps = list(range(1, len(objective_values) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=objective_values,
        mode='lines+markers',
        name='Objective Function',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='<b>Step:</b> %{x}<br><b>Objective Function:</b> %{y:.6f}<extra></extra>'
    ))

    best_value = min(objective_values)
    fig.add_hline(
        y=best_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Best: {best_value:.6f}",
        annotation_position="top right"
    )

    if len(objective_values) > 10:
        window_size = max(5, len(objective_values) // 20)
        moving_avg = []
        for i in range(len(objective_values)):
            start_idx = max(0, i - window_size + 1)
            avg = sum(objective_values[start_idx:i + 1]) / (i - start_idx + 1)
            moving_avg.append(avg)

        fig.add_trace(go.Scatter(
            x=steps,
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window_size} steps)',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='<b>Step:</b> %{x}<br><b>Moving Avg:</b> %{y:.6f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text="ATAT SQS Optimization Progress",
            font=dict(size=24, family="Arial Black")  # Increased from 18
        ),
        xaxis_title="Optimization Step",
        yaxis_title="Objective Function Value",
        hovermode='x unified',
        height=650,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16)
        ),
        font=dict(size=20, family="Arial"),
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            title_font=dict(size=20, family="Arial Black"),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            title_font=dict(size=20, family="Arial Black"),
            tickfont=dict(size=16)
        )
    )

    return fig


def analyze_convergence(objective_values):
    if len(objective_values) < 10:
        return {
            "Convergence Status": "❓ Insufficient Data",
            "Final Trend": "Need more steps",
            "Stability": "Unknown",
            "Recommendation": "Run longer optimization"
        }

    total_steps = len(objective_values)
    final_quarter = objective_values[-total_steps // 4:]

    final_std = np.std(final_quarter)
    final_mean = np.mean(final_quarter)

    total_improvement = objective_values[-1] - objective_values[0]

    recent_improvement = final_quarter[-1] - final_quarter[0] if len(final_quarter) > 1 else 0

    if final_std < 0.001 and abs(recent_improvement) < 0.001:
        convergence_status = "✅ Converged"
        recommendation = "Optimization complete - SQS ready for use"
    elif recent_improvement < -0.001:
        convergence_status = "⚠️ Improving"
        recommendation = "Still improving - consider running longer"
    elif final_std > 0.01:
        convergence_status = "🔄 Fluctuating"
        recommendation = "Consider adjusting parameters or running longer"
    else:
        convergence_status = "📊 Stable"
        recommendation = "Appears stable - likely converged"

    if final_std < 0.0001:
        stability = "Very Stable"
    elif final_std < 0.001:
        stability = "Stable"
    elif final_std < 0.01:
        stability = "Moderately Stable"
    else:
        stability = "Unstable"

    if abs(recent_improvement) < 0.0001:
        final_trend = "Flat (converged)"
    elif recent_improvement < -0.001:
        final_trend = "Improving"
    elif recent_improvement > 0.001:
        final_trend = "Worsening"
    else:
        final_trend = "Minor fluctuations"

    return {
        "Convergence Status": convergence_status,
        "Final Trend": final_trend,
        "Stability": stability,
        "Recent Std Dev": f"{final_std:.6f}",
        "Total Improvement": f"{total_improvement:.6f}",
        "Recent Improvement": f"{recent_improvement:.6f}",
        "Recommendation": recommendation
    }


def validate_mcsqs_log(log_content):
    try:
        lines = log_content.strip().split('\n')

        if len(lines) < 3:
            return False, "File too short - not a valid mcsqs.log"

        has_objective = any("Objective_function=" in line for line in lines)
        has_correlations = any("Correlations_mismatch=" in line for line in lines)

        if not has_objective:
            return False, "No 'Objective_function=' lines found"

        objective_count = sum(1 for line in lines if "Objective_function=" in line)

        return True, f"Valid mcsqs.log with {objective_count} optimization steps"

    except Exception as e:
        return False, f"Error parsing log file: {str(e)}"


def calculate_atat_valid_concentrations(achievable_concentrations, use_sublattice_mode,
                                        chem_symbols, transformation_matrix, primitive_structure):
    nx, ny, nz = transformation_matrix[0, 0], transformation_matrix[1, 1], transformation_matrix[2, 2]
    supercell_multiplicity = nx * ny * nz

    num_primitive_sites = len(primitive_structure)
    total_supercell_sites = num_primitive_sites * supercell_multiplicity

    if not use_sublattice_mode:
        target_atoms_by_element = {}
        for element, fraction in achievable_concentrations.items():
            target_atoms_by_element[element] = int(round(fraction * total_supercell_sites))

        site_assignments = find_optimal_atom_distribution_global(
            target_atoms_by_element, num_primitive_sites, supercell_multiplicity
        )

    else:
        site_assignments = find_optimal_atom_distribution_sublattice(
            achievable_concentrations, chem_symbols, transformation_matrix, primitive_structure
        )

    final_site_assignments = {}
    for site_idx, atom_counts in site_assignments.items():
        concentrations = {}
        total_atoms_at_site = sum(atom_counts.values())

        if total_atoms_at_site != supercell_multiplicity:
            print(f"Warning: Site {site_idx} has {total_atoms_at_site} atoms, expected {supercell_multiplicity}")

        for element, count in atom_counts.items():
            concentrations[element] = count / supercell_multiplicity

        final_site_assignments[site_idx] = concentrations

    return final_site_assignments


def find_optimal_atom_distribution_global(target_atoms_by_element, num_primitive_sites, supercell_multiplicity):
    site_assignments = {}
    for site_idx in range(num_primitive_sites):
        site_assignments[site_idx] = {element: 0 for element in target_atoms_by_element.keys()}

    for element, total_target_atoms in target_atoms_by_element.items():
        remaining_atoms = total_target_atoms

        atoms_per_site = remaining_atoms // num_primitive_sites
        for site_idx in range(num_primitive_sites):
            site_assignments[site_idx][element] = atoms_per_site
            remaining_atoms -= atoms_per_site

        for site_idx in range(remaining_atoms):
            site_assignments[site_idx][element] += 1

    for site_idx in range(num_primitive_sites):
        total_atoms_at_site = sum(site_assignments[site_idx].values())

        if total_atoms_at_site > supercell_multiplicity:
            excess = total_atoms_at_site - supercell_multiplicity
            elements_sorted = sorted(site_assignments[site_idx].items(), key=lambda x: x[1], reverse=True)

            for element, count in elements_sorted:
                if excess <= 0:
                    break
                reduction = min(excess, count)
                site_assignments[site_idx][element] -= reduction
                excess -= reduction

        elif total_atoms_at_site < supercell_multiplicity:
            deficit = supercell_multiplicity - total_atoms_at_site
            elements_sorted = sorted(site_assignments[site_idx].items(), key=lambda x: x[1], reverse=True)

            if elements_sorted:
                most_abundant_element = elements_sorted[0][0]
                site_assignments[site_idx][most_abundant_element] += deficit

    return site_assignments


def find_optimal_atom_distribution_sublattice(achievable_concentrations, chem_symbols,
                                              transformation_matrix, primitive_structure):
    nx, ny, nz = transformation_matrix[0, 0], transformation_matrix[1, 1], transformation_matrix[2, 2]
    supercell_multiplicity = nx * ny * nz

    sublattice_mapping = {}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    unique_combinations = {}
    for site_idx, site_elements in enumerate(chem_symbols):
        if len(site_elements) > 1:
            elements_signature = frozenset(sorted(site_elements))
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

    site_assignments = {}

    for sublattice_letter, sublattice_concentrations in achievable_concentrations.items():
        if sublattice_letter in sublattice_mapping:
            site_indices = sublattice_mapping[sublattice_letter]['site_indices']
            num_sites_in_sublattice = len(site_indices)

            total_sublattice_sites_in_supercell = num_sites_in_sublattice * supercell_multiplicity
            target_atoms_by_element = {}

            for element, fraction in sublattice_concentrations.items():
                target_atoms_by_element[element] = int(round(fraction * total_sublattice_sites_in_supercell))

            sublattice_site_assignments = find_optimal_atom_distribution_global(
                target_atoms_by_element, num_sites_in_sublattice, supercell_multiplicity
            )

            for local_idx, global_idx in enumerate(site_indices):
                site_assignments[global_idx] = sublattice_site_assignments[local_idx]

    for site_idx, site_elements in enumerate(chem_symbols):
        if len(site_elements) == 1 and site_idx not in site_assignments:
            element = site_elements[0]
            site_assignments[site_idx] = {element: supercell_multiplicity}

    return site_assignments


def generate_atat_rndstr_content_corrected(structure, achievable_concentrations, use_sublattice_mode,
                                           chem_symbols, transformation_matrix):
    lattice = structure.lattice
    max_param = max(lattice.a, lattice.b, lattice.c) if max(lattice.a, lattice.b, lattice.c) > 0 else 1
    lines = [
        f"{lattice.a / max_param:.6f} {lattice.b / max_param:.6f} {lattice.c / max_param:.6f} {lattice.alpha:.2f} {lattice.beta:.2f} {lattice.gamma:.2f}",
        "1 0 0", "0 1 0", "0 0 1"
    ]

    if use_sublattice_mode:
        site_assignments = calculate_atat_valid_concentrations(
            achievable_concentrations, use_sublattice_mode, chem_symbols,
            transformation_matrix, structure
        )
        for i, site in enumerate(structure):
            coords = site.frac_coords
            coord_str = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
            conc_parts = []
            if i in site_assignments:
                for element, conc in sorted(site_assignments[i].items()):
                    if conc > 1e-6:
                        conc_parts.append(f"{element}={conc:.6f}")
            lines.append(f"{coord_str} {','.join(conc_parts)}")
    else:
        conc_parts = []
        for element, conc in sorted(achievable_concentrations.items()):
            if conc > 1e-6:
                conc_parts.append(f"{element}={conc:.6f}")
        conc_str = ",".join(conc_parts)

        for site in structure:
            coords = site.frac_coords
            coord_str = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
            lines.append(f"{coord_str} {conc_str}")

    return "\n".join(lines)


def generate_atat_sqscell_content(nx, ny, nz):
    lines = []
    lines.append("1")
    lines.append("")

    lines.append(f"{nx} 0 0")
    lines.append(f"0 {ny} 0")
    lines.append(f"0 0 {nz}")

    return "\n".join(lines)


def generate_atat_input_files(structure, target_concentrations, transformation_matrix,
                              use_sublattice_mode, chem_symbols, nx, ny, nz,
                              pair_cutoff, triplet_cutoff, quadruplet_cutoff, total_atoms):
    return generate_atat_input_files_corrected(
        structure, target_concentrations, transformation_matrix,
        use_sublattice_mode, chem_symbols, nx, ny, nz,
        pair_cutoff, triplet_cutoff, quadruplet_cutoff, total_atoms
    )


def verify_atat_concentrations(site_assignments, transformation_matrix, primitive_structure,
                               target_total_atoms_by_element):
    nx, ny, nz = transformation_matrix[0, 0], transformation_matrix[1, 1], transformation_matrix[2, 2]
    supercell_multiplicity = nx * ny * nz

    total_atoms_produced = {}

    for site_idx, concentrations in site_assignments.items():
        for element, concentration in concentrations.items():
            if element not in total_atoms_produced:
                total_atoms_produced[element] = 0

            total_atoms_produced[element] += concentration * supercell_multiplicity

    print("Verification:")
    print("Element | Target | Produced | Match")
    print("-" * 35)
    for element in target_total_atoms_by_element:
        target = target_total_atoms_by_element[element]
        produced = total_atoms_produced.get(element, 0)
        match = "✓" if abs(target - produced) < 0.001 else "✗"
        print(f"{element:7} | {target:6.1f} | {produced:8.1f} | {match}")


def display_atat_concentration_info(supercell_multiplicity):
    st.info(f"""
    **ATAT Concentration Requirements for {supercell_multiplicity}×replication:**

    Valid concentrations must be multiples of 1/{supercell_multiplicity} = {1 / supercell_multiplicity:.6f}

    **Valid values:** {', '.join([f'{i}/{supercell_multiplicity} = {i / supercell_multiplicity:.6f}' for i in range(supercell_multiplicity + 1)])}

    Each concentration represents the fraction of {supercell_multiplicity} atoms at that site position.
    """)


def convert_achievable_sublattice_to_site_assignments(structure, achievable_concentrations, chem_symbols):
    site_assignments = {}

    sublattice_mapping = {}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    unique_combinations = {}
    for site_idx, site_elements in enumerate(chem_symbols):
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

    for sublattice_letter, concentrations in achievable_concentrations.items():
        if sublattice_letter in sublattice_mapping:
            site_indices = sublattice_mapping[sublattice_letter]['site_indices']
            for site_idx in site_indices:
                site_assignments[site_idx] = concentrations.copy()

    for site_idx, site_elements in enumerate(chem_symbols):
        if len(site_elements) == 1 and site_idx not in site_assignments:
            element = site_elements[0]
            site_assignments[site_idx] = {element: 1.0}

    return site_assignments


def integrate_atat_option():
    st.markdown(
        """
        <hr style="border: none; height: 8px; background: linear-gradient(45deg, #ff6600, #ff9933); border-radius: 8px; margin: 30px 0;">
        """,
        unsafe_allow_html=True
    )

    st.title("🛠️ ATAT SQS Input File Generator")
    st.markdown("**Generate input files for ATAT mcsqs to create Special Quasirandom Structures**")
    st.info("""
    This tool generates `rndstr.in` and `sqscell.out` files that can be used with the ATAT (Alloy Theoretic Automated Toolkit) 
    to create Special Quasirandom Structures. Use the same composition settings as ICET, but generate files for external ATAT usage.

    **Key Features:**
    - ✅ **Valid Concentrations**: Each site shows concentrations that represent integer atom counts
    - ✅ **Supercell Aware**: Accounts for site replication in supercell expansion  
    - ✅ **ICET Compatible**: Uses same achievable concentration calculations as ICET
    - ✅ **Both Modes**: Supports global and sublattice-specific composition control
    """)

    render_atat_sqs_section()


def render_site_sublattice_selector_fixed(working_structure, all_sites, unique_sites, supercell_multiplicity,
                                          stable_key="default"):
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #8B0000; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )
    st.subheader("🔵3️⃣ Step 3: Configure Sublattices (Unique Wyckoff Positions Only)")

    st.info(f"""
    **Sublattice Mode - Wyckoff Position Control:**
    - Each supercell (for all 3 directions) replication creates {supercell_multiplicity} copies per primitive site. 
    Only unique Wyckoff positions are shown below. Settings automatically apply to all equivalent sites. Concentration constraints are per Wyckoff position.
    """)

    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
        'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Vac'
    ]

    target_concentrations = {}
    chem_symbols = [[] for _ in range(len(working_structure))]
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    unique_wyckoff_groups = {}

    for site_info in unique_sites:
        key = (site_info['element'], site_info['wyckoff_letter'])
        if key not in unique_wyckoff_groups:
            unique_wyckoff_groups[key] = []
        unique_wyckoff_groups[key].append(site_info)

    sublattice_data = []
    temp_sublattice_index = 0

    for group_key, site_infos in unique_wyckoff_groups.items():
        element, wyckoff_letter = group_key

        # Calculate total multiplicity for this group
        total_multiplicity = sum(site_info['multiplicity'] for site_info in site_infos)
        all_equivalent_indices = []
        for site_info in site_infos:
            all_equivalent_indices.extend(site_info['equivalent_indices'])

        if temp_sublattice_index < len(sublattice_letters):
            sublattice_letter = sublattice_letters[temp_sublattice_index]

            sublattice_data.append({
                'sublattice_letter': sublattice_letter,
                'element': element,
                'wyckoff_letter': wyckoff_letter,
                'all_equivalent_indices': all_equivalent_indices,
                'total_multiplicity': total_multiplicity,
                'atoms_per_wyckoff_in_supercell': total_multiplicity * supercell_multiplicity,
                'min_concentration_step': 1.0 / (total_multiplicity * supercell_multiplicity)
            })

            temp_sublattice_index += 1

    if sublattice_data:
        tab_names = [f"Sublattice {data['sublattice_letter']}" for data in sublattice_data]
        tabs = st.tabs(tab_names)
        
        css = '''
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.15rem !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            background-color: #f0f4ff !important;
            border-radius: 12px !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease !important;
            border: none !important;
            color: #1e3a8a !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button:hover {
            background-color: #dbe5ff !important;
            cursor: pointer;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #e0e7ff !important;
            color: #1e3a8a !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
        
            /* Added underline (thicker) */
            border-bottom: 4px solid #1e3a8a !important;
            border-radius: 12px 12px 0 0 !important; /* keep rounded only on top */
        }
        
        .stTabs [data-baseweb="tab-list"] button:focus {
            outline: none !important;
        }
        </style>
        '''


        st.markdown(css, unsafe_allow_html=True)
        for tab_idx, (tab, data) in enumerate(zip(tabs, sublattice_data)):
            with tab:
                sublattice_letter = data['sublattice_letter']
                element = data['element']
                wyckoff_letter = data['wyckoff_letter']
                all_equivalent_indices = data['all_equivalent_indices']
                total_multiplicity = data['total_multiplicity']
                atoms_per_wyckoff_in_supercell = data['atoms_per_wyckoff_in_supercell']
                min_concentration_step = data['min_concentration_step']

                st.write(f"### Sublattice {sublattice_letter}: {element} @ {wyckoff_letter} positions")
                st.write(f"**Multiplicity:** {total_multiplicity} (affects {len(all_equivalent_indices)} sites)")
                st.write(f"**Atoms per supercell:** {atoms_per_wyckoff_in_supercell}")

                # Show constraint information
                st.info(f"**Concentration constraints for this Wyckoff position:**\n"
                        f"- Total atoms in supercell: {atoms_per_wyckoff_in_supercell}\n"
                        f"- Minimum concentration step: {min_concentration_step:.6f}\n")
                col_elem, col_conc = st.columns([1, 2])
                element_key = f"{stable_key}_sublattice_{sublattice_letter}_elements_v2"
                with col_elem:
                    current_elements = [element]

                    selected_elements = st.multiselect(
                        f"Elements for sublattice {sublattice_letter}:",
                        options=common_elements,
                        default=current_elements,
                        key=element_key,
                        help=f"Select elements that can occupy {wyckoff_letter} positions"
                    )
                    if len(selected_elements) < 1:
                        st.warning(f"Select at least 1 element for sublattice {sublattice_letter}")
                        continue

                with col_conc:
                    st.write(f"**Set concentrations for sublattice {sublattice_letter}:**")

                    sublattice_concentrations = {}
                    remaining = 1.0

                    for i, elem in enumerate(selected_elements[:-1]):
                        slider_key = f"{stable_key}_sublattice_{sublattice_letter}_{elem}_frac_v2"
                        frac_val = st.slider(
                            f"**{elem} fraction:**",
                            min_value=0.0,
                            max_value=remaining,
                            value=min(
                                int(atoms_per_wyckoff_in_supercell / len(selected_elements)) * min_concentration_step,
                                remaining),
                            step=min_concentration_step,
                            format="%.6f",
                            key=slider_key
                        )
                        sublattice_concentrations[elem] = frac_val
                        remaining -= frac_val

                    if selected_elements:
                        last_elem = selected_elements[-1]
                        sublattice_concentrations[last_elem] = max(0.0, remaining)
                        st.write(f"**{last_elem}: {sublattice_concentrations[last_elem]:.6f}** (automatic)")

                    total_frac = sum(sublattice_concentrations.values())
                    if abs(total_frac - 1.0) > 1e-6:
                        st.error(f"Total fraction = {total_frac:.6f}, should be 1.0")
                    else:
                        st.success(f"✅ Total fraction = {total_frac:.6f}")

                    st.write("**Resulting atom counts:**")
                    for elem, frac in sublattice_concentrations.items():
                        atom_count = frac * atoms_per_wyckoff_in_supercell
                        st.write(f"- {elem}: {atom_count:.1f} atoms")

                if len(selected_elements) >= 1:
                    target_concentrations[sublattice_letter] = sublattice_concentrations

                    for site_idx in all_equivalent_indices:
                        chem_symbols[site_idx] = selected_elements.copy()

    return chem_symbols, target_concentrations, None


import streamlit as st
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.build import make_supercell


# Assuming helpers module contains necessary functions like pymatgen_to_ase, get_unique_sites, get_all_sites, structure_preview, sqs_visualization
# from helpers import * # Removed the above line as I don't have access to your helpers.py,
# but the structure_preview and sqs_visualization are critical for context.


def display_sublattice_preview_fixed(target_concentrations, chem_symbols, transformation_matrix, working_structure,
                                     unique_sites):
    try:
        nx, ny, nz = transformation_matrix[0, 0], transformation_matrix[1, 1], transformation_matrix[2, 2]
        supercell_multiplicity = nx * ny * nz
        total_primitive_sites = len(working_structure)

        st.write("**Sublattice Configuration Preview:**")
        sublattice_display_info = {}

        unique_site_map = {}
        for u_site_info in unique_sites:
            for idx in u_site_info['equivalent_indices']:
                unique_site_map[idx] = u_site_info

        for sublattice_letter, concentrations in target_concentrations.items():
            first_associated_original_site_idx = -1
            for i, site_elements_at_idx in enumerate(chem_symbols):
                if len(site_elements_at_idx) > 1 and set(site_elements_at_idx) == set(concentrations.keys()):
                    first_associated_original_site_idx = i
                    break

            if first_associated_original_site_idx != -1:
                u_site_info = unique_site_map.get(first_associated_original_site_idx)
                if u_site_info:
                    total_atoms_in_this_sublattice_supercell = u_site_info['multiplicity'] * supercell_multiplicity

                    conc_parts = []
                    for element, frac in sorted(concentrations.items()):
                        if frac > 1e-6:
                            conc_parts.append(f"{element}={frac:.6f}")

                    sublattice_display_info[sublattice_letter] = {
                        "Wyckoff Position": f"{u_site_info['element']} @ {u_site_info['wyckoff_letter']}",
                        "Multiplicity": u_site_info['multiplicity'],
                        "Supercell Atoms": total_atoms_in_this_sublattice_supercell,
                        "Element Assignments": ", ".join(conc_parts)
                    }

        preview_data = []
        for s_letter in sorted(sublattice_display_info.keys()):
            preview_data.append({"Sublattice": s_letter, **sublattice_display_info[s_letter]})

        if preview_data:
            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True)
        else:
            st.warning("No mixed sublattices configured yet.")
        pure_sites_data = []
        for u_site_info in unique_sites:
            original_site_index = u_site_info['equivalent_indices'][0]
            if original_site_index < len(chem_symbols) and len(chem_symbols[original_site_index]) == 1:
                is_part_of_mixed_sublattice = False
                for target_concs_dict in target_concentrations.values():
                    if set(chem_symbols[original_site_index]) == set(target_concs_dict.keys()):
                        is_part_of_mixed_sublattice = True
                        break

                if not is_part_of_mixed_sublattice:
                    pure_element = chem_symbols[original_site_index][0]
                    pure_sites_data.append({
                        "Wyckoff Position": f"{pure_element} @ {u_site_info['wyckoff_letter']}",
                        "Multiplicity": u_site_info['multiplicity'],
                        "Supercell Atoms": u_site_info['multiplicity'] * supercell_multiplicity,
                        "Status": f"Pure {pure_element} (unchanged)"
                    })

        if pure_sites_data:
            st.write("**Pure (Unchanged) Sites:**")
            pure_df = pd.DataFrame(pure_sites_data)
            st.dataframe(pure_df, use_container_width=True)

        st.markdown(
            """
            <hr style="border: none; height: 3px; background-color: #555; border-radius: 8px; margin: 20px 0;">
            """,
            unsafe_allow_html=True
        )
        st.write("#### **Overall Expected Element Distribution in Supercell:**")

        total_element_counts = {}
        for i in range(total_primitive_sites):
            u_site_info = unique_site_map.get(i)
            if not u_site_info:
                continue

            site_multiplicity_in_primitive = 1
            site_multiplicity_in_supercell = site_multiplicity_in_primitive * supercell_multiplicity
            site_elements_list = chem_symbols[i]

            current_site_composition = {}
            if len(site_elements_list) == 1:  # Pure site.
                is_now_mixed = False
                for t_concs in target_concentrations.values():
                    if set(site_elements_list) == set(t_concs.keys()):
                        current_site_composition = t_concs
                        is_now_mixed = True
                        break
                if not is_now_mixed:
                    current_site_composition = {site_elements_list[0]: 1.0}  # Pure element, 100% occupancy
            else:
                found_sublattice_config = False
                for t_concs in target_concentrations.values():
                    if set(site_elements_list) == set(t_concs.keys()):
                        current_site_composition = t_concs
                        found_sublattice_config = True
                        break
                if not found_sublattice_config:
                    # st.warning(
                    #     f"Warning: Site {i} has multiple elements {site_elements_list} but no matching sublattice configuration. Defaulting to original fractional occupancies if available.")
                    if working_structure[i].is_ordered:
                        current_site_composition = {working_structure[i].specie.symbol: 1.0}
                    else:
                        current_site_composition = {str(sp): occ for sp, occ in working_structure[i].species.items()}

            for element, fraction in current_site_composition.items():
                atom_count_at_this_site_in_supercell = fraction * site_multiplicity_in_supercell
                total_element_counts[element] = total_element_counts.get(element,
                                                                         0) + atom_count_at_this_site_in_supercell

        total_atoms_in_overall_supercell = sum(total_element_counts.values())

        ase_atoms = pymatgen_to_ase(working_structure)
        supercell_preview_atoms = make_supercell(ase_atoms, transformation_matrix)
        expected_total_atoms = len(supercell_preview_atoms)

        if abs(total_atoms_in_overall_supercell - expected_total_atoms) > 1e-6 and total_atoms_in_overall_supercell > 0:
            correction_factor = expected_total_atoms / total_atoms_in_overall_supercell
            for elem in total_element_counts:
                total_element_counts[elem] *= correction_factor
            total_atoms_in_overall_supercell = expected_total_atoms

        if total_element_counts:
            cols = st.columns(min(len(total_element_counts), 4))
            for i, (elem, count) in enumerate(sorted(total_element_counts.items())):
                percentage = (
                                     count / total_atoms_in_overall_supercell) * 100 if total_atoms_in_overall_supercell > 0 else 0
                with cols[i % len(cols)]:
                    if percentage >= 80:
                        color = "#2E4057"
                    elif percentage >= 60:
                        color = "#4A6741"
                    elif percentage >= 40:
                        color = "#6B73FF"
                    elif percentage >= 25:
                        color = "#FF8C00"
                    elif percentage >= 15:
                        color = "#4ECDC4"  # Teal
                    elif percentage >= 10:
                        color = "#45B7D1"  # Blue
                    elif percentage >= 5:
                        color = "#96CEB4"  # Green
                    elif percentage >= 2:
                        color = "#FECA57"  # Yellow
                    elif percentage >= 1:
                        color = "#DDA0DD"  # Plum
                    else:
                        color = "#D3D3D3"  # Light Gray

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {color}, {color}CC);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        margin: 10px 0;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                        border: 2px solid rgba(255,255,255,0.2);
                    ">
                        <h1 style="
                            color: white;
                            font-size: 3em;
                            margin: 0;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                            font-weight: bold;
                        ">{elem}</h1>
                        <h2 style="
                            color: white;
                            font-size: 2em;
                            margin: 10px 0 0 0;
                            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                        ">{percentage:.1f}%</h2>
                        <p style="
                            color: white;
                            font-size: 1.8em;
                            margin: 5px 0 0 0;
                            opacity: 0.9;
                        ">{int(round(count, 0))} atoms</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.write(f"**Total expected atoms in supercell:** {int(total_atoms_in_overall_supercell)}")

    except Exception as e:
        st.error(f"Error creating sublattice preview: {e}")


def calculate_achievable_concentrations_sublattice_fixed(target_concentrations, chem_symbols, transformation_matrix,
                                                         working_structure, unique_sites):
    nx, ny, nz = transformation_matrix[0, 0], transformation_matrix[1, 1], transformation_matrix[2, 2]
    supercell_multiplicity = nx * ny * nz

    achievable_concentrations = {}
    adjustment_info = []

    for sublattice_letter, target_fractions in target_concentrations.items():
        sublattice_multiplicity = 0

        sublattice_site_indices = []
        for i, site_elements in enumerate(chem_symbols):
            if len(site_elements) >= 1 and set(site_elements) == set(target_fractions.keys()):
                sublattice_site_indices.append(i)

        for site_info in unique_sites:
            if any(idx in site_info['equivalent_indices'] for idx in sublattice_site_indices):
                relevant_indices = [idx for idx in site_info['equivalent_indices'] if idx in sublattice_site_indices]
                sublattice_multiplicity += len(relevant_indices)

        if sublattice_multiplicity == 0:
            continue

        total_atoms_in_sublattice_supercell = sublattice_multiplicity * supercell_multiplicity

        achievable_fractions = {}
        target_atom_counts = {}

        for element, target_frac in target_fractions.items():
            target_atoms = target_frac * total_atoms_in_sublattice_supercell
            rounded_atoms = round(target_atoms)
            target_atom_counts[element] = rounded_atoms

        total_target = sum(target_atom_counts.values())
        if total_target != total_atoms_in_sublattice_supercell:
            diff = total_atoms_in_sublattice_supercell - total_target
            largest_element = max(target_atom_counts.keys(), key=lambda x: target_atom_counts[x])
            target_atom_counts[largest_element] += diff

        for element, atom_count in target_atom_counts.items():
            achievable_fractions[element] = atom_count / total_atoms_in_sublattice_supercell

            original_frac = target_fractions[element]
            if abs(original_frac - achievable_fractions[element]) > 1e-6:
                adjustment_info.append({
                    'Sublattice': sublattice_letter,
                    'Element': element,
                    'Target (%)': f"{original_frac * 100:.3f}",
                    'Achievable (%)': f"{achievable_fractions[element] * 100:.3f}",
                    'Atom Count': atom_count
                })

        achievable_concentrations[sublattice_letter] = achievable_fractions

    return achievable_concentrations, adjustment_info


def calculate_achievable_concentrations(target_concentrations, total_atoms):
    achievable_concentrations = {}
    achievable_counts = {}

    target_atom_counts = {}
    for element, frac in target_concentrations.items():
        target_atom_counts[element] = frac * total_atoms

    rounded_counts = {}
    for element, count in target_atom_counts.items():
        rounded_counts[element] = round(count)

    total_rounded = sum(rounded_counts.values())
    diff = total_atoms - total_rounded

    if diff != 0:
        errors = {}
        for element in target_atom_counts:
            original = target_atom_counts[element]
            rounded = rounded_counts[element]
            errors[element] = abs(original - rounded)

        sorted_elements = sorted(errors.keys(), key=lambda x: errors[x], reverse=True)

        for i in range(abs(diff)):
            element = sorted_elements[i % len(sorted_elements)]
            if diff > 0:
                rounded_counts[element] += 1
            else:
                rounded_counts[element] = max(0, rounded_counts[element] - 1)

    for element, count in rounded_counts.items():
        achievable_concentrations[element] = count / total_atoms
        achievable_counts[element] = count

    return achievable_concentrations, achievable_counts


def generate_atat_rndstr_content_corrected(structure, achievable_concentrations, use_sublattice_mode, chem_symbols,
                                           transformation_matrix):
    lattice = structure.lattice
    max_param = max(lattice.a, lattice.b, lattice.c) if max(lattice.a, lattice.b, lattice.c) > 0 else 1
    lines = [
        f"{lattice.a / max_param:.6f} {lattice.b / max_param:.6f} {lattice.c / max_param:.6f} {lattice.alpha:.2f} {lattice.beta:.2f} {lattice.gamma:.2f}",
        "1 0 0", "0 1 0", "0 0 1"
    ]

    if use_sublattice_mode:
        for i, site in enumerate(structure):
            coords = site.frac_coords
            coord_str = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"

            site_elements = chem_symbols[i] if i < len(chem_symbols) else []
            # print(f"Site {i}: site_elements = {site_elements}")
            # print(f"Site {i}: achievable_concentrations = {achievable_concentrations}")
            if len(site_elements) >= 1:
                conc_parts = []
                for sublattice_letter, sublattice_concentrations in achievable_concentrations.items():
                    if set(site_elements) == set(sublattice_concentrations.keys()):
                        for element, conc in sorted(sublattice_concentrations.items()):
                            if conc > 1e-6:
                                conc_parts.append(f"{element}={conc:.6f}")
                        break

                if conc_parts:
                    lines.append(f"{coord_str} {','.join(conc_parts)}")
                else:
                    lines.append(f"{coord_str} {site.specie.symbol}")
            else:
                if site.is_ordered:
                    lines.append(f"{coord_str} {site.specie.symbol}")
                else:
                    conc_parts = []
                    for sp, occ in site.species.items():
                        if occ > 1e-6:
                            conc_parts.append(f"{sp.symbol}={occ:.6f}")
                    lines.append(f"{coord_str} {','.join(conc_parts)}")
    else:
        conc_parts = []
        for element, conc in sorted(achievable_concentrations.items()):
            if conc > 1e-6:
                conc_parts.append(f"{element}={conc:.6f}")
        conc_str = ",".join(conc_parts)

        for site in structure:
            coords = site.frac_coords
            coord_str = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
            lines.append(f"{coord_str} {conc_str}")

    return "\n".join(lines)


def generate_atat_sqscell_content(nx, ny, nz):
    lines = []
    lines.append("1")
    lines.append("")
    lines.append(f"{nx} 0 0")
    lines.append(f"0 {ny} 0")
    lines.append(f"0 0 {nz}")
    return "\n".join(lines)


def generate_atat_command_sequence(pair_cutoff, triplet_cutoff, quadruplet_cutoff, total_atoms):
    commands = []
    commands.append(
        "# Step 1: Generate cluster information, -noe: not including empty cluster, -nop: not including point cluster(s)")
    cluster_cmd = f"corrdump -l=rndstr.in -ro -noe -nop -clus -2={round(pair_cutoff, 3)}"

    if triplet_cutoff is not None:
        cluster_cmd += f" -3={triplet_cutoff}"
    if quadruplet_cutoff is not None:
        cluster_cmd += f" -4={quadruplet_cutoff}"

    commands.append(cluster_cmd)
    commands.append("")
    commands.append("# Step 2: (Optional) View generated clusters")
    commands.append("getclus")
    commands.append("")
    commands.append("# Step 3: Generate SQS structure")
    commands.append("# Option A: Use predefined supercell from sqscell.out")
    commands.append("mcsqs -rc")
    commands.append("")
    commands.append(
        "# Option B: Specify number of atoms directly (will search for the most randomized supercell - distorts the original cell shape)")
    commands.append(f"mcsqs -n {total_atoms}")
    commands.append("")
    commands.append("# Step 4: (Optional) Parallel execution for faster results")
    commands.append("mcsqs -rc -ip=1 &")
    commands.append("mcsqs -rc -ip=2 &")
    commands.append("mcsqs -rc -ip=3 &")
    commands.append("wait")
    commands.append("")
    commands.append("# Step 5: Monitor progress and convert results using the following sections in this app")

    return "\n".join(commands)


def generate_atat_input_files_corrected(structure, target_concentrations, transformation_matrix,
                                        use_sublattice_mode, chem_symbols, nx, ny, nz,
                                        pair_cutoff, triplet_cutoff, quadruplet_cutoff, total_atoms):
    if use_sublattice_mode:
        achievable_concentrations = target_concentrations
        adjustment_info = []
    else:
        achievable_concentrations, achievable_counts = calculate_achievable_concentrations(
            target_concentrations, total_atoms
        )
        adjustment_info = []
        for element in target_concentrations:
            if abs(target_concentrations[element] - achievable_concentrations[element]) > 1e-6:
                adjustment_info.append({
                    'Element': element,
                    'Target (%)': f"{target_concentrations[element] * 100:.3f}",
                    'Achievable (%)': f"{achievable_concentrations[element] * 100:.3f}",
                    'Atom Count': achievable_counts[element]
                })

    rndstr_content = generate_atat_rndstr_content_corrected(
        structure, achievable_concentrations, use_sublattice_mode,
        chem_symbols, transformation_matrix
    )

    sqscell_content = generate_atat_sqscell_content(nx, ny, nz)
    atat_commands = generate_atat_command_sequence(pair_cutoff, triplet_cutoff, quadruplet_cutoff, total_atoms)

    return rndstr_content, sqscell_content, atat_commands, achievable_concentrations, adjustment_info


def parse_mcsqs_progress_csv(csv_content):
    from io import StringIO

    try:
        df = pd.read_csv(StringIO(csv_content))

        required_columns = ['Minute', 'Objective_Function']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        minutes = df['Minute'].tolist()
        objective_values = df['Objective_Function'].tolist()

        additional_data = {}
        optional_columns = ['Step_Count', 'First_Correlation', 'Total_Correlations', 'Status', 'Timestamp']
        for col in optional_columns:
            if col in df.columns:
                additional_data[col] = df[col].tolist()

        return minutes, objective_values, additional_data

    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")


def create_optimization_plot_csv(minutes, objective_values, additional_data=None):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=minutes,
        y=objective_values,
        mode='lines',
        name='Objective Function',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4, color='#1f77b4'),
        hovertemplate='<b>Minute:</b> %{x}<br><b>Objective Function:</b> %{y:.6f}<extra></extra>'
    ))

    best_value = min(objective_values)
    fig.add_hline(
        y=best_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Best: {best_value:.6f}",
        annotation_position="top right"
    )

    if len(objective_values) > 5:
        window_size = max(3, len(objective_values) // 10)
        moving_avg = []
        for i in range(len(objective_values)):
            start_idx = max(0, i - window_size + 1)
            avg = sum(objective_values[start_idx:i + 1]) / (i - start_idx + 1)
            moving_avg.append(avg)

        fig.add_trace(go.Scatter(
            x=minutes,
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window_size} points)',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='<b>Minute:</b> %{x}<br><b>Moving Avg:</b> %{y:.6f}<extra></extra>'
        ))

    if additional_data and 'Step_Count' in additional_data:
        fig.add_trace(go.Scatter(
            x=minutes,
            y=additional_data['Step_Count'],
            mode='lines',
            name='Step Count',
            line=dict(color='green', width=2),
            # marker=dict(size=3, color='green'),
            yaxis='y2',
            hovertemplate='<b>Minute:</b> %{x}<br><b>Steps:</b> %{y}<extra></extra>'
        ))

        fig.update_layout(
            yaxis2=dict(
                title="Step Count",
                overlaying='y',
                side='right',
                title_font=dict(size=18, family="Arial Black", color='green'),
                tickfont=dict(color='green')
            )
        )

    fig.update_layout(
        title=dict(
            text="ATAT SQS Optimization Progress (CSV Data)",
            font=dict(size=24, family="Arial Black")
        ),
        xaxis_title="Time (Minutes)",
        yaxis_title="Objective Function Value",
        hovermode='x unified',
        height=650,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16)
        ),
        font=dict(size=20, family="Arial"),
        plot_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            title_font=dict(size=20, family="Arial Black"),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=1,
            title_font=dict(size=20, family="Arial Black"),
            tickfont=dict(size=16)
        )
    )

    return fig


def analyze_convergence_csv(minutes, objective_values, additional_data=None):
    import numpy as np

    if len(objective_values) < 5:
        return {
            "Convergence Status": "❓ Insufficient Data",
            "Final Trend": "Need more data points",
            "Stability": "Unknown",
            "Recommendation": "Continue optimization"
        }

    total_minutes = len(minutes)
    final_quarter = objective_values[-total_minutes // 4:] if total_minutes >= 4 else objective_values[-2:]

    final_std = np.std(final_quarter)
    final_mean = np.mean(final_quarter)

    total_improvement = objective_values[-1] - objective_values[0]

    recent_improvement = final_quarter[-1] - final_quarter[0] if len(final_quarter) > 1 else 0

    if final_std < 0.001 and abs(recent_improvement) < 0.001:
        convergence_status = "✅ Converged"
        recommendation = "Optimization complete - SQS ready for use"
    elif recent_improvement < -0.001:
        convergence_status = "⚠️ Improving"
        recommendation = "Still improving - consider running longer"
    elif final_std > 0.01:
        convergence_status = "🔄 Fluctuating"
        recommendation = "Consider adjusting parameters or running longer"
    else:
        convergence_status = "📊 Stable"
        recommendation = "Appears stable - likely converged"

    if final_std < 0.0001:
        stability = "Very Stable"
    elif final_std < 0.001:
        stability = "Stable"
    elif final_std < 0.01:
        stability = "Moderately Stable"
    else:
        stability = "Unstable"

    if abs(recent_improvement) < 0.0001:
        final_trend = "Flat (converged)"
    elif recent_improvement < -0.001:
        final_trend = "Improving"
    elif recent_improvement > 0.001:
        final_trend = "Worsening"
    else:
        final_trend = "Minor fluctuations"

    if len(minutes) > 1:
        time_span = minutes[-1] - minutes[0]
        improvement_rate = total_improvement / time_span if time_span > 0 else 0
    else:
        improvement_rate = 0

    result = {
        # "Convergence Status": convergence_status,
        # "Final Trend": final_trend,
        "Stability": stability,
        "Recent Std Dev": f"{final_std:.6f}",
        "Total Improvement": f"{total_improvement:.6f}",
        "Recent Improvement": f"{recent_improvement:.6f}",
        "Improvement Rate": f"{improvement_rate:.6f} per minute",
        "Total Runtime": f"{minutes[-1] - minutes[0]:.0f} minutes" if len(minutes) > 1 else "N/A",
        # "Recommendation": recommendation
    }

    if additional_data and 'Step_Count' in additional_data:
        step_counts = additional_data['Step_Count']
        if step_counts:
            result["Final Step Count"] = str(step_counts[-1])
            if len(step_counts) > 1:
                total_steps = step_counts[-1] - step_counts[0]
                result["Steps per Minute"] = f"{total_steps / (minutes[-1] - minutes[0]):.1f}" if minutes[-1] > minutes[
                    0] else "N/A"

    return result


def validate_mcsqs_progress_csv(csv_content):
    from io import StringIO

    try:

        df = pd.read_csv(StringIO(csv_content))

        required_columns = ['Minute', 'Objective_Function']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"

        if len(df) == 0:
            return False, "CSV file is empty"

        try:
            df['Minute'] = pd.to_numeric(df['Minute'])
            df['Objective_Function'] = pd.to_numeric(df['Objective_Function'])
        except ValueError as e:
            return False, f"Invalid numeric data: {str(e)}"

        if df['Minute'].min() < 0:
            return False, "Minute values cannot be negative"

        data_points = len(df)
        time_span = df['Minute'].max() - df['Minute'].min()

        return True, f"Valid CSV with {data_points} data points over {time_span:.0f} minutes"

    except Exception as e:
        return False, f"Error parsing CSV file: {str(e)}"


def render_extended_optimization_analysis_tab():
    st.write("**Upload optimization files to analyze progress:**")
    log_tab, csv_tab, parallel_tab, correlation_tab, tab_para = st.tabs([
        "📊 mcsqs.log Analysis",
        "📈 mcsqs_progress.csv Analysis",
        "🔄 Parallel Runs Analysis",
        "🔗 bestcorr.out Analysis",
        "📊 Parallel CSV Analysis",
    ])
    with tab_para:
        render_parallel_csv_analysis_tab()
    with log_tab:
        st.write("**Upload mcsqs.log file to analyze optimization progress:**")
        uploaded_mcsqs_log = st.file_uploader(
            "Upload mcsqs.log file:",
            type=['log', 'txt'],
            help="Upload the mcsqs.log file generated during ATAT optimization",
            key="mcsqs_log_uploader"
        )

        if uploaded_mcsqs_log is not None:
            try:
                log_content = uploaded_mcsqs_log.read().decode('utf-8')
                objective_values, correlation_data = parse_mcsqs_log(log_content)

                if not objective_values:
                    st.warning("No objective function values found in the log file.")
                    st.info("Please ensure the file contains 'Objective_function=' lines.")
                    return

                st.success(f"✅ Successfully parsed {len(objective_values)} optimization steps!")

                fig = create_optimization_plot(objective_values)
                st.plotly_chart(fig, use_container_width=True)

                col_stats1, col_stats2, col_stats3 = st.columns(3)

                with col_stats1:
                    st.metric("Total Steps", len(objective_values))
                    st.metric("Initial Value", f"{objective_values[0]:.6f}")

                with col_stats2:
                    final_value = objective_values[-1]
                    improvement = final_value - objective_values[0]
                    st.metric("Final Value", f"{final_value:.6f}")
                    st.metric("Total Improvement", f"{improvement:.6f}")

                with col_stats3:
                    best_value = min(objective_values)
                    best_step = objective_values.index(best_value) + 1
                    st.metric("Best Value", f"{best_value:.6f}")
                    st.metric("Best at Step", best_step)

                # Convergence analysis
                # st.subheader("📈 Convergence Analysis")

                # Calculate convergence metrics
                # convergence_info = analyze_convergence(objective_values)

                # col_conv_info1, col_conv_info2 = st.columns(2)

                # with col_conv_info1:
                #    st.write("**Convergence Metrics:**")
                #    for key, value in convergence_info.items():
                #        st.write(f"- **{key}:** {value}")

                # with col_conv_info2:
                # Create convergence assessment
                #    if convergence_info['Convergence Status'] == "✅ Converged":
                #        st.success("🎯 **Optimization Status: CONVERGED**")
                #        st.info("The objective function has stabilized. This SQS is ready for use.")
                #    elif convergence_info['Convergence Status'] == "⚠️ Improving":
                #        st.warning("📈 **Optimization Status: STILL IMPROVING**")
                #        st.info("Consider running ATAT longer for better results.")
                #    else:
                #        st.warning("🔄 **Optimization Status: FLUCTUATING**")
                #        st.info("The optimization may need more steps or different parameters.")

                with st.expander("📊 Detailed Optimization Data", expanded=False):
                    # Create DataFrame with step numbers and objective values
                    data_df = pd.DataFrame({
                        'Step': range(1, len(objective_values) + 1),
                        'Objective Function': objective_values,
                        'Improvement': [0.0] + [objective_values[i] - objective_values[i - 1]
                                                for i in range(1, len(objective_values))]
                    })

                    st.dataframe(data_df, use_container_width=True)

                    csv_data = data_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Optimization Data (CSV)",
                        data=csv_data,
                        file_name=f"mcsqs_optimization_analysis.csv",
                        mime="text/csv",
                        key="download_optimization_csv"
                    )

            except UnicodeDecodeError:
                st.error("Error reading log file. Please ensure the file is a text file with UTF-8 encoding.")
            except Exception as e:
                st.error(f"Error processing mcsqs.log file: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")

    with csv_tab:
        st.write("**Upload mcsqs_progress.csv file for time-based analysis:**")
        uploaded_progress_csv = st.file_uploader(
            "Upload mcsqs_progress.csv file:",
            type=['csv'],
            help="Upload the mcsqs_progress.csv file for time-based optimization analysis",
            key="mcsqs_progress_csv_uploader"
        )

        if uploaded_progress_csv is not None:
            try:
                csv_content = uploaded_progress_csv.read().decode('utf-8')
                is_valid, validation_message = validate_mcsqs_progress_csv(csv_content)

                if not is_valid:
                    st.error(f"Invalid CSV file: {validation_message}")
                    st.info("Please ensure the CSV has 'Minute' and 'Objective_Function' columns.")
                    return

                st.success(f"✅ Valid CSV file: {validation_message}")
                minutes, objective_values, additional_data = parse_mcsqs_progress_csv(csv_content)

                st.success(f"✅ Successfully parsed {len(objective_values)} data points!")
                fig = create_optimization_plot_csv(minutes, objective_values, additional_data)
                st.plotly_chart(fig, use_container_width=True)

                col_stats1, col_stats2, col_stats3 = st.columns(3)

                with col_stats1:
                    st.metric("Total Data Points", len(objective_values))
                    st.metric("Initial Value", f"{objective_values[0]:.6f}")

                with col_stats2:
                    final_value = objective_values[-1]
                    improvement = final_value - objective_values[0]
                    st.metric("Final Value", f"{final_value:.6f}")
                    st.metric("Total Improvement", f"{improvement:.6f}")

                with col_stats3:
                    best_value = min(objective_values)
                    runtime = minutes[-1] - minutes[0] if len(minutes) > 1 else 0
                    st.metric("Best Value", f"{best_value:.6f}")
                    st.metric("Runtime", f"{runtime:.0f} min")

                st.subheader("📈 Time-Based Convergence Analysis")

                convergence_info = analyze_convergence_csv(minutes, objective_values, additional_data)

                col_conv_info1, col_conv_info2 = st.columns(2)

                with col_conv_info1:
                    st.write("**Convergence Metrics:**")
                    for key, value in convergence_info.items():
                        st.write(f"- **{key}:** {value}")

                # with col_conv_info2:

                # if convergence_info['Convergence Status'] == "✅ Converged":
                #    st.success("🎯 **Optimization Status: CONVERGED**")
                #    st.info("The objective function has stabilized. This SQS is ready for use.")
                # if convergence_info['Convergence Status'] == "⚠️ Improving":
                #     st.warning("📈 **Optimization Status: STILL IMPROVING**")
                #     st.info("Consider running ATAT longer for better results.")
                # else:
                #     st.warning("🔄 **Optimization Status: FLUCTUATING**")
                #     st.info("The optimization may need more time or different parameters.")

                if additional_data:
                    with st.expander("📊 Additional Data Analysis", expanded=False):
                        data_dict = {
                            'Minute': minutes,
                            'Objective_Function': objective_values
                        }

                        for key, values in additional_data.items():
                            if len(values) == len(minutes):
                                data_dict[key] = values

                        df_display = pd.DataFrame(data_dict)
                        st.dataframe(df_display, use_container_width=True)

                        csv_data = df_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Data (CSV)",
                            data=csv_data,
                            file_name=f"mcsqs_progress_analysis.csv",
                            mime="text/csv",
                            key="download_progress_csv"
                        )

            # with st.expander("📊 Raw CSV Data", expanded=False):

            #     display_data = {
            #         'Minute': minutes,
            #         'Objective_Function': objective_values
            #     }

            #     for key, values in additional_data.items():
            #         if len(values) == len(minutes):
            #             display_data[key] = values

            #     df_raw = pd.DataFrame(display_data)
            #     st.dataframe(df_raw, use_container_width=True)

            except UnicodeDecodeError:
                st.error("Error reading CSV file. Please ensure the file is a text file with UTF-8 encoding.")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")

    with parallel_tab:
        st.write("**Upload multiple log files from parallel ATAT runs:**")
        st.info("""
        Upload multiple mcsqs.log files (e.g., mcsqs1.log, mcsqs2.log, mcsqs3.log) from parallel execution to compare their performance.
        This analysis will show you which parallel run achieved the best objective function.
        """)

        uploaded_parallel_logs = st.file_uploader(
            "Upload multiple mcsqs log files:",
            type=['log', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple log files from parallel ATAT runs (e.g., mcsqs1.log, mcsqs2.log, etc.)",
            key="parallel_logs_uploader"
        )

        if uploaded_parallel_logs and len(uploaded_parallel_logs) > 1:
            try:
                parallel_results = []
                for i, log_file in enumerate(uploaded_parallel_logs):
                    try:
                        log_content = log_file.read().decode('utf-8')
                        objective_values, _ = parse_mcsqs_log(log_content)

                        if objective_values:
                            final_objective = objective_values[-1]
                            best_objective = min(objective_values)
                            total_steps = len(objective_values)
                            improvement = final_objective - objective_values[0] if len(objective_values) > 1 else 0

                            run_match = re.search(r'(\d+)', log_file.name)
                            run_number = int(run_match.group(1)) if run_match else i + 1

                            parallel_results.append({
                                'File': log_file.name,
                                # 'Run': i + 1,
                                'Run': run_number,
                                'BestSQS': f'bestsqs{run_number}.out',
                                'Final_Objective': final_objective,
                                'Best_Objective': best_objective,
                                'Total_Steps': total_steps,
                                'Total_Improvement': improvement,
                                'Objective_Values': objective_values
                            })
                        else:
                            st.warning(f"No objective function values found in {log_file.name}")

                    except Exception as e:
                        st.warning(f"Error processing {log_file.name}: {str(e)}")

                if parallel_results:
                    st.success(f"✅ Successfully processed {len(parallel_results)} parallel runs!")

                    # Calculate statistics
                    final_objectives = [r['Final_Objective'] for r in parallel_results]
                    best_objectives = [r['Best_Objective'] for r in parallel_results]

                    min_final = min(final_objectives)
                    max_final = max(final_objectives)
                    avg_final = sum(final_objectives) / len(final_objectives)

                    min_best = min(best_objectives)
                    max_best = max(best_objectives)
                    avg_best = sum(best_objectives) / len(best_objectives)

                    # Find best and worst runs
                    best_run_idx = final_objectives.index(min_final)
                    worst_run_idx = final_objectives.index(max_final)
                    best_run = parallel_results[best_run_idx]
                    worst_run = parallel_results[worst_run_idx]

                    # Display summary statistics
                    st.subheader("📊 Parallel Runs Summary")

                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                    with col_stat1:
                        st.metric("Total Runs", len(parallel_results))
                        st.metric("Best Final Objective", f"{min_final:.6f}")

                    with col_stat2:
                        st.metric("Worst Final Objective", f"{max_final:.6f}")
                        st.metric("Average Final Objective", f"{avg_final:.6f}")

                    with col_stat3:
                        st.metric("Best Overall Objective", f"{min_best:.6f}")
                        st.metric("Standard Deviation", f"{np.std(final_objectives):.6f}")

                    with col_stat4:
                        st.metric("Objective Range", f"{max_final - min_final:.6f}")
                        st.metric("Best Run", f"{best_run['File']}")

                    st.subheader("🏆 Best vs Worst Runs")

                    col_best, col_worst = st.columns(2)

                    with col_best:
                        st.success(f"**🥇 Best Performing Run:\t\t{best_run['File']}** ({best_run['BestSQS']})")
                        st.write(f"**File:** {best_run['File']}")
                        # st.write(f"**Final Objective:** {best_run['Final_Objective']:.6f}")
                        st.write(f"**Best Objective:** {best_run['Best_Objective']:.6f}")
                        st.write(f"**Total Steps:** {best_run['Total_Steps']}")
                        st.write(f"**Total Improvement:** {best_run['Total_Improvement']:.6f}")

                    with col_worst:
                        st.error(f"**🥉 Worst Performing Run:\t\t{worst_run['File']}** ({worst_run['BestSQS']})")
                        st.write(f"**File:** {worst_run['File']}")
                        # st.write(f"**Final Objective:** {worst_run['Final_Objective']:.6f}")
                        st.write(f"**Best Objective:** {worst_run['Best_Objective']:.6f}")
                        st.write(f"**Total Steps:** {worst_run['Total_Steps']}")
                        st.write(f"**Total Improvement:** {worst_run['Total_Improvement']:.6f}")

                    st.subheader("📋 Detailed Comparison")

                    comparison_data = []
                    for result in parallel_results:
                        comparison_data.append({
                            "File": result['File'],
                            "Run #": result['Run'],
                            # "Final Objective": f"{result['Final_Objective']:.6f}",
                            "Best Objective": f"{result['Best_Objective']:.6f}",
                            "Total Steps": result['Total_Steps'],
                            "Improvement": f"{result['Total_Improvement']:.6f}",
                            "Performance": "🥇 Best" if result == best_run else "🥉 Worst" if result == worst_run else "✅ OK"
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    st.subheader("📈 Combined Optimization Progress")

                    import plotly.graph_objects as go

                    fig = go.Figure()

                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

                    for i, result in enumerate(parallel_results):
                        color = colors[i % len(colors)]
                        line_style = dict(color=color,
                                          width=3 if result == best_run else 2 if result == worst_run else 1)

                        name_suffix = " (Best)" if result == best_run else " (Worst)" if result == worst_run else ""

                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(result['Objective_Values']) + 1)),
                            y=result['Objective_Values'],
                            mode='lines',
                            name=f"{result['File']}{name_suffix}",
                            line=line_style,
                            hovertemplate=f'<b>{result["File"]}</b><br>Step: %{{x}}<br>Objective: %{{y:.6f}}<extra></extra>'
                        ))

                    fig.update_layout(
                        title=dict(
                            text="Parallel Runs Comparison",
                            font=dict(size=24, family="Arial Black")
                        ),
                        xaxis_title="Optimization Step",
                        yaxis_title="Objective Function Value",
                        hovermode='x unified',
                        height=650,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            font=dict(size=16)
                        ),
                        font=dict(size=20, family="Arial"),
                        xaxis=dict(
                            title_font=dict(size=20, family="Arial Black"),
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict(
                            title_font=dict(size=20, family="Arial Black"),
                            tickfont=dict(size=16)
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📥 Download Results")

                    summary_data = {
                        'Metric': [
                            'Number of Runs',
                            'Best Final Objective',
                            'Worst Final Objective',
                            'Average Final Objective',
                            'Standard Deviation',
                            'Best Overall Objective',
                            'Best Run File',
                            'Worst Run File'
                        ],
                        'Value': [
                            len(parallel_results),
                            f"{min_final:.6f}",
                            f"{max_final:.6f}",
                            f"{avg_final:.6f}",
                            f"{np.std(final_objectives):.6f}",
                            f"{min_best:.6f}",
                            best_run['File'],
                            worst_run['File']
                        ]
                    }

                    summary_df = pd.DataFrame(summary_data)
                    detailed_df = pd.DataFrame(comparison_data)

                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        summary_csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Summary (CSV)",
                            data=summary_csv,
                            file_name="parallel_runs_summary.csv",
                            mime="text/csv",
                            key="download_parallel_summary"
                        )

                    with col_dl2:
                        detailed_csv = detailed_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detailed Results (CSV)",
                            data=detailed_csv,
                            file_name="parallel_runs_detailed.csv",
                            mime="text/csv",
                            key="download_parallel_detailed"
                        )

                else:
                    st.error("No valid log files could be processed.")

            except Exception as e:
                st.error(f"Error processing parallel log files: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")

        elif uploaded_parallel_logs and len(uploaded_parallel_logs) == 1:
            st.warning("Please upload at least 2 log files to compare parallel runs.")
            st.info("For single file analysis, use the 'mcsqs.log Analysis' tab instead.")
        else:
            st.info("Upload multiple log files from parallel ATAT runs to see comparison analysis.")
    with correlation_tab:
        render_correlation_analysis_tab()


def generate_atat_monitor_script(results, use_atom_count=False, parallel_runs=1, pair_cutoff=1.1, triplet_cutoff=None,
                                 quadruplet_cutoff=None,max_param=1.0,time_limit_minutes=None):
    if use_atom_count:
        mcsqs_base_cmd = f"mcsqs -n {results['total_atoms']}"
    else:
        mcsqs_base_cmd = "mcsqs -rc"

    corrdump_cmd = f"corrdump -l=rndstr.in -ro -noe -nop -clus -2={pair_cutoff}"
    if triplet_cutoff:
        corrdump_cmd += f" -3={triplet_cutoff}"
    if quadruplet_cutoff:
        corrdump_cmd += f" -4={quadruplet_cutoff}"

    if parallel_runs > 1:
        mcsqs_commands = []
        for i in range(1, parallel_runs + 1):
            if time_limit_minutes:
                mcsqs_commands.append(
                    f"timeout {time_limit_minutes * 60}s {mcsqs_base_cmd} -ip={i} > mcsqs{i}.log 2>&1 || true &")
            else:
                mcsqs_commands.append(f"{mcsqs_base_cmd} -ip={i} > mcsqs{i}.log 2>&1 &")
        mcsqs_execution = "\n".join(mcsqs_commands)
        log_file = "mcsqs1.log"
        mcsqs_display_cmd = f"{mcsqs_base_cmd} -ip=1 & {mcsqs_base_cmd} -ip=2 & ... (parallel execution)"
        progress_file = "mcsqs_parallel_progress.csv"

        monitoring_function = f'''start_parallel_monitoring_process() {{
   local output_file="$1"
   local minute=0

   echo "Monitor started for {parallel_runs} parallel runs. Waiting for 5 seconds to allow mcsqs to initialize..."
   sleep 5

   header="Minute,Timestamp"
   for i in $(seq 1 {parallel_runs}); do
       header="$header,Run${{i}}_Steps,Run${{i}}_Objective,Run${{i}}_Status"
   done
   header="$header,Best_Overall_Objective,Best_Run"
   echo "$header" > "$output_file"

   echo "----------------------------------------"
   echo "Monitoring {parallel_runs} parallel MCSQS runs every minute"
   echo "Log files: mcsqs1.log, mcsqs2.log, ..., mcsqs{parallel_runs}.log"
   echo "----------------------------------------"

   while true; do
       minute=$((minute + 1))
       local current_time=$(date +"%m/%d/%Y %H:%M")

       row_data="$minute,$current_time"
       best_objective=""
       best_run=""
       any_running=false

       for i in $(seq 1 {parallel_runs}); do
           local log_file="mcsqs${{i}}.log"
           local objective="N/A"
           local step_count="0"
           local status="STOPPED"

           if pgrep -f "mcsqs.*-ip=${{i}}" > /dev/null; then
               status="RUNNING"
               any_running=true
           fi

           if [ -f "$log_file" ]; then
               objective=$(extract_latest_objective "$log_file")
               step_count=$(extract_latest_step "$log_file")
               objective=${{objective:-"N/A"}}
               step_count=${{step_count:-"0"}}
           fi

           row_data="$row_data,$step_count,$objective,$status"

           if [ "$objective" != "N/A" ] && [ -n "$objective" ]; then
               if [ -z "$best_objective" ] || awk "BEGIN {{exit !($objective < $best_objective)}}" 2>/dev/null; then
                   best_objective="$objective"
                   best_run="Run$i"
               fi
           fi
       done

       best_objective=${{best_objective:-"N/A"}}
       best_run=${{best_run:-"N/A"}}
       row_data="$row_data,$best_objective,$best_run"

       echo "$row_data" >> "$output_file"

       printf "Minute %3d | Active runs: " "$minute"
        for i in $(seq 1 {parallel_runs}); do
            if pgrep -f "mcsqs.*-ip=${{i}}" > /dev/null; then
                printf "R%d " "$i"
            else
                printf "%s " "--"  
            fi
        done
        printf "| Best: %s (%s)\\n" "$best_objective" "$best_run"

       if [ "$any_running" = false ]; then
           echo "All parallel runs stopped. Collecting final data..."
           break
       fi

       sleep 60
   done

   echo "----------------------------------------"
   echo "Parallel monitoring process finished."
}}'''

        monitor_call = "start_parallel_monitoring_process \"$PROGRESS_FILE\" &"

    else:
        if time_limit_minutes:
            mcsqs_execution = f"timeout {time_limit_minutes * 60}s {mcsqs_base_cmd} > \"$LOG_FILE\" 2>&1 || true &"
        else:
            if time_limit_minutes:
                mcsqs_execution = f"timeout {time_limit_minutes * 60}s {mcsqs_base_cmd} > \"$LOG_FILE\" 2>&1 || true &"
            else:
                mcsqs_execution = f"{mcsqs_base_cmd} > \"$LOG_FILE\" 2>&1 &"
        log_file = "mcsqs.log"
        mcsqs_display_cmd = mcsqs_base_cmd
        progress_file = "mcsqs_progress.csv"

        monitoring_function = '''start_monitoring_process() {
   local log_file="$1"
   local output_file="$2"
   local minute=0

   echo "Monitor started. Waiting for 5 seconds to allow mcsqs to initialize..."
   sleep 5

   echo "Minute,Timestamp,Step_Count,Objective_Function,First_Correlation,Total_Correlations,Status" > "$output_file"
   echo "----------------------------------------"
   echo "Initial read complete. Now monitoring in 1-minute intervals."

   while true; do
       minute=$((minute + 1))
       local current_time=$(date +"%m/%d/%Y %H:%M")
       local status

       if is_mcsqs_running; then
           status="RUNNING"
       else
           status="STOPPED"
       fi

       local objective=$(extract_latest_objective "$log_file")
       local step_count=$(extract_latest_step "$log_file")
       local correlation=$(extract_latest_correlation "$log_file")
       local corr_count=$(count_correlations "$log_file")

       objective=${objective:-"N/A"}
       step_count=${step_count:-"0"}
       correlation=${correlation:-"N/A"}
       corr_count=${corr_count:-"0"}

       echo "$minute,$current_time,$step_count,$objective,$correlation,$corr_count,$status" >> "$output_file"

       printf "Minute %3d | Steps: %6s | Objective: %12s | Status: %s\\n" \\
              "$minute" "$step_count" "$objective" "$status"

       if [ "$status" = "STOPPED" ]; then
           echo "MCSQS process stopped. Monitoring will collect final data before exiting."
           break
       fi

       sleep 60
   done

   echo "----------------------------------------"
   echo "Monitoring process finished."
}'''

        monitor_call = "start_monitoring_process \"$LOG_FILE\" \"$PROGRESS_FILE\" &"

    script_content = f'''#!/bin/bash

# ATAT MCSQS Run with Integrated Progress Monitoring
# Auto-generated script with embedded file creation
# Generated configuration: {results['structure_name']}, {results['supercell_size']}, {results['total_atoms']} atoms

# --- Configuration ---
LOG_FILE="{log_file}"
PROGRESS_FILE="{progress_file}"
DEFAULT_MCSQS_ARGS="{mcsqs_base_cmd.split('mcsqs ')[1]}"
{"TIME_LIMIT_MINUTES=" + str(time_limit_minutes) if time_limit_minutes else "TIME_LIMIT_MINUTES=0"}
TIME_LIMIT_SECONDS=$((TIME_LIMIT_MINUTES * 60))

# --- Auto-generate ATAT Input Files ---
create_input_files() {{
   echo "Creating ATAT input files..."

   cat > rndstr.in << 'EOF'
{results['rndstr_content']}
EOF

   cat > sqscell.out << 'EOF'
{results['sqscell_content']}
EOF

   echo "✅ Input files created: rndstr.in, sqscell.out"
}}

# --- Monitoring Functions ---

extract_latest_objective() {{
   grep "Objective_function=" "$1" | tail -1 | sed 's/.*= *//' 2>/dev/null || echo ""
}}

extract_latest_step() {{
   grep -c "Objective_function=" "$1" 2>/dev/null || echo "0"
}}

extract_latest_correlation() {{
   grep "Correlations_mismatch=" "$1" | tail -1 | sed 's/.*= *//' | awk '{{print $1}}' 2>/dev/null || echo ""
}}

count_correlations() {{
   grep "Correlations_mismatch=" "$1" | tail -1 | awk -F'\\t' '{{print NF-1}}' 2>/dev/null || echo "0"
}}

is_mcsqs_running() {{
   pgrep -f "mcsqs" > /dev/null
   return $?
}}

convert_bestsqs_to_poscar() {{
    local bestsqs_file="$1"
    local poscar_file="$2"
    
    if [ ! -f "$bestsqs_file" ]; then
        echo "⚠️  Warning: $bestsqs_file not found"
        return 1
    fi
    
    echo "🔄 Converting $bestsqs_file to $poscar_file..."
    
    python3 - "$bestsqs_file" "$poscar_file" << 'PYEOF'
import sys
import numpy as np

def parse_bestsqs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    A = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
    B = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])
    
    A_scaled = A * {max_param:.6f}
    final_lattice = np.dot(B, A_scaled)
    
    atoms = []
    for i in range(6, len(lines)):
        line = lines[i].strip()
        if line:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z, element = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]
                if element.lower() in ['vac', "'vac", 'vacancy', 'x']:
                    continue
                cart_pos = np.dot([x, y, z], A_scaled)
                atoms.append((element, cart_pos))
    
    return final_lattice, atoms

def write_poscar(lattice, atoms, filename, comment):
    from collections import defaultdict
    
    element_groups = defaultdict(list)
    for element, pos in atoms:
        element_groups[element].append(pos)
    
    atomic_weights = {{
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81, 'C': 12.01, 'N': 14.01, 'O': 16.00,
        'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07,
        'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00,
        'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63,
        'As': 74.92, 'Se': 78.96, 'Br': 79.90, 'Kr': 83.80, 'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22,
        'Nb': 92.91, 'Mo': 95.96, 'Tc': 98.00, 'Ru': 101.1, 'Rh': 102.9, 'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4,
        'In': 114.8, 'Sn': 118.7, 'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3, 'Cs': 132.9, 'Ba': 137.3,
        'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2, 'Pm': 145.0, 'Sm': 150.4, 'Eu': 152.0, 'Gd': 157.3,
        'Tb': 158.9, 'Dy': 162.5, 'Ho': 164.9, 'Er': 167.3, 'Tm': 168.9, 'Yb': 173.0, 'Lu': 175.0, 'Hf': 178.5,
        'Ta': 180.9, 'W': 183.8, 'Re': 186.2, 'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1, 'Au': 197.0, 'Hg': 200.6,
        'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0,
        'Ac': 227.0, 'Th': 232.0, 'Pa': 231.0, 'U': 238.0, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0,
        'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 'Lr': 262.0
    }}
    
    elements = sorted(element_groups.keys(), key=lambda x: atomic_weights.get(x, 999.0))
    
    with open(filename, 'w') as f:
        f.write(f'{{comment}}\\n')
        f.write('1.0\\n')
        
        for vec in lattice:
            f.write(f'  {{vec[0]:15.9f}} {{vec[1]:15.9f}} {{vec[2]:15.9f}}\\n')
        
        f.write(' '.join(elements) + '\\n')
        f.write(' '.join(str(len(element_groups[el])) for el in elements) + '\\n')
        
        f.write('Direct\\n')
        inv_lattice = np.linalg.inv(lattice)
        for element in elements:
            for cart_pos in element_groups[element]:
                frac_pos = np.dot(cart_pos, inv_lattice)
                f.write(f'  {{frac_pos[0]:15.9f}} {{frac_pos[1]:15.9f}} {{frac_pos[2]:15.9f}}\\n')

try:
    import sys
    bestsqs_file = sys.argv[1] if len(sys.argv) > 1 else "$bestsqs_file"
    poscar_file = sys.argv[2] if len(sys.argv) > 2 else "$poscar_file"
    
    comment = f"SQS from {{bestsqs_file}}"
    lattice, atoms = parse_bestsqs(bestsqs_file)
    write_poscar(lattice, atoms, poscar_file, comment)
    print(f"✅ Successfully converted {{bestsqs_file}} to {{poscar_file}}")
except Exception as e:
    print(f"❌ Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
    
    return $?
}}

{monitoring_function}

# --- Main Script Logic ---

check_prerequisites() {{
   echo "Checking prerequisites..."

   create_input_files

   echo "Generating clusters with corrdump..."
   echo "Command: {corrdump_cmd}"
   {corrdump_cmd}
   if [ $? -ne 0 ]; then
       echo "ERROR: corrdump command failed!"
       exit 1
   fi
   echo "✅ Clusters generated successfully."
   echo "✅ All prerequisites satisfied."
}}
cleanup() {{
   echo ""
   echo "=========================================="
   echo "🛑 Interrupt signal received or process completed"
   echo "=========================================="
   
   echo "🧹 Stopping MCSQS processes..."
   if [ -n "$MCSQS_PID" ]; then kill "$MCSQS_PID" 2>/dev/null; fi
   if [ -n "$MONITOR_PID" ]; then kill "$MONITOR_PID" 2>/dev/null; fi
   if [ -n "$TIMER_PID" ]; then kill "$TIMER_PID" 2>/dev/null; fi
   pkill -9 -f "mcsqs" 2>/dev/null || true
   sleep 2
   
   echo ""
   echo "=========================================="
   echo "📄 Converting bestsqs*.out files to POSCAR format..."
   echo "=========================================="
   
   found_files=0
   best_run=""
   best_objective=""
   
   if [ -f "$PROGRESS_FILE" ]; then
       last_line=$(tail -1 "$PROGRESS_FILE")
       {"best_objective=$(echo \"$last_line\" | cut -d',' -f$((3 + 3 * " + str(parallel_runs) + ")))" if parallel_runs > 1 else "best_objective=$(echo \"$last_line\" | cut -d',' -f4)"}
       {"best_run=$(echo \"$last_line\" | cut -d',' -f$((4 + 3 * " + str(parallel_runs) + ")) | sed 's/Run//')" if parallel_runs > 1 else "best_run=\"1\""}
   fi
   
   for outfile in bestsqs*.out; do
       if [ -f "$outfile" ]; then
           found_files=1
           basename="${{outfile%.out}}"
           poscar_file="${{basename}}_POSCAR"
           
           if convert_bestsqs_to_poscar "$outfile" "$poscar_file"; then
               echo "  ✅ $outfile → $poscar_file"
           else
               echo "  ❌ Failed to convert $outfile"
           fi
       fi
   done
   
   if [ $found_files -eq 0 ]; then
       echo "  ⚠️  No bestsqs*.out files found"
   else
       if [ -n "$best_run" ] && [ -n "$best_objective" ]; then
           if [ {parallel_runs} -gt 1 ]; then
               best_poscar="bestsqs${{best_run}}_POSCAR"
           else
               best_poscar="bestsqs_POSCAR"
           fi
           
           if [ -f "$best_poscar" ]; then
               cp "$best_poscar" "POSCAR_best_overall"
               echo ""
               echo "🏆 Best structure (objective: $best_objective) saved as POSCAR_best_overall"
               if [ {parallel_runs} -gt 1 ]; then
                   echo "    Source: Run $best_run (bestsqs${{best_run}}.out)"
               else
                   echo "    Source: bestsqs.out"
               fi
           else
               echo ""
               echo "⚠️  Could not find best POSCAR file: $best_poscar"
           fi
       else
           echo ""
           echo "⚠️  Could not determine best structure (no progress data found)"
       fi
       
       echo ""
       echo "=========================================="
       echo "✅ Conversion complete!"
       echo "=========================================="
   fi
   
   exit 0
}}

trap cleanup SIGINT SIGTERM

# --- Execution ---
echo "================================================"
echo "    ATAT MCSQS with Integrated Monitoring"
echo "================================================"
echo "Configuration:"
echo "  - Structure: {results['structure_name']}"
echo "  - Supercell: {results['supercell_size']} ({results['total_atoms']} atoms)"
echo "  - Parallel runs: {parallel_runs}"
echo "  - Command: {mcsqs_display_cmd}"
{"echo \"  - Time limit: $TIME_LIMIT_MINUTES minutes\"" if time_limit_minutes else "echo \"  - Time limit: None (manual stop)\""}
echo "  - Log file: $LOG_FILE"
echo "  - Progress file: $PROGRESS_FILE"
echo "================================================"

check_prerequisites

rm -f "$LOG_FILE" "$PROGRESS_FILE" mcsqs*.log
echo ""
echo "Starting ATAT MCSQS optimization and progress monitor..."

{mcsqs_execution}
MCSQS_PID=$!



{monitor_call}
MONITOR_PID=$!

echo "✅ MCSQS started"
echo "✅ Monitor started (PID: $MONITOR_PID)"
echo ""
echo "Real-time progress logged to: $PROGRESS_FILE"
if [ $TIME_LIMIT_MINUTES -gt 0 ]; then
    echo "⏱️  Will auto-stop after $TIME_LIMIT_MINUTES minutes"
    echo "Press Ctrl+C to stop earlier and auto-convert to POSCAR."
else
    echo "Press Ctrl+C to stop optimization and auto-convert to POSCAR."
fi
echo "================================================"

{"wait" if parallel_runs > 1 else "wait $MCSQS_PID"}
MCSQS_EXIT_CODE=$?

echo ""
if [ $MCSQS_EXIT_CODE -eq 124 ]; then
    echo "⏱️  Time limit reached ($TIME_LIMIT_MINUTES minutes). MCSQS stopped automatically."
else
    echo "MCSQS process finished with exit code: $MCSQS_EXIT_CODE."
fi

echo "Allowing monitor to capture final data..."
sleep 5

kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "=========================================="
echo "🔄 Converting bestsqs files to POSCAR..."
echo "=========================================="

found_files=0
best_run=""
best_objective=""

if [ -f "$PROGRESS_FILE" ]; then
    last_line=$(tail -1 "$PROGRESS_FILE")
    {"best_objective=$(echo \"$last_line\" | cut -d',' -f$((3 + 3 * " + str(parallel_runs) + ")))" if parallel_runs > 1 else "best_objective=$(echo \"$last_line\" | cut -d',' -f4)"}
    {"best_run=$(echo \"$last_line\" | cut -d',' -f$((4 + 3 * " + str(parallel_runs) + ")) | sed 's/Run//')" if parallel_runs > 1 else "best_run=\"1\""}
fi

for outfile in bestsqs*.out; do
    if [ -f "$outfile" ]; then
        found_files=1
        basename="${{outfile%.out}}"
        poscar_file="${{basename}}_POSCAR"
        
        if convert_bestsqs_to_poscar "$outfile" "$poscar_file"; then
            echo "  ✅ $outfile → $poscar_file"
        else
            echo "  ❌ Failed to convert $outfile"
        fi
    fi
done

if [ $found_files -eq 0 ]; then
    echo "  ⚠️  No bestsqs*.out files found"
else
    if [ -n "$best_run" ] && [ -n "$best_objective" ]; then
        if [ {parallel_runs} -gt 1 ]; then
            best_poscar="bestsqs${{best_run}}_POSCAR"
        else
            best_poscar="bestsqs_POSCAR"
        fi
        
        if [ -f "$best_poscar" ]; then
            cp "$best_poscar" "POSCAR_best_overall"
            echo ""
            echo "🏆 Best structure (objective: $best_objective) saved as POSCAR_best_overall"
            if [ {parallel_runs} -gt 1 ]; then
                echo "    Source: Run $best_run (bestsqs${{best_run}}.out)"
            else
                echo "    Source: bestsqs.out"
            fi
        else
            echo ""
            echo "⚠️  Could not find best POSCAR file: $best_poscar"
        fi
    else
        echo ""
        echo "⚠️  Could not determine best structure (no progress data found)"
    fi
fi

echo ""
echo "================================================"
echo "              Optimization Complete"
echo "================================================"


if [ -f "$PROGRESS_FILE" ]; then
   echo "Progress Summary:"
   echo "  - Total monitoring time:   ~$(tail -1 "$PROGRESS_FILE" | cut -d',' -f1) minutes"
   {"echo \"  - Best overall objective:  $(tail -1 \"$PROGRESS_FILE\" | cut -d',' -f$((3 + 3 * " + str(parallel_runs) + ")))\"" if parallel_runs > 1 else "echo \"  - Final objective function: $(tail -1 \"$PROGRESS_FILE\" | cut -d',' -f4)\""}
fi

echo "================================================"
'''

    return script_content


def render_monitor_script_section(results):
    st.subheader("🖥️ Advanced Monitoring Script")
    st.info("""
    **Download a complete monitoring script** that:
    - ✅ **Auto-creates** rndstr.in and sqscell.out files
    - ✅ **Runs corrdump** automatically with your settings
    - ✅ **Executes mcsqs** with real-time monitoring
    - ✅ **Generates CSV progress** data every minute
    - ✅ **Supports parallel execution** for faster results
    - ✅ **Creates POSCAR from bestsqs.out** automatically
    """)

    # Configuration options
    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        st.write("**MCSQS Execution Mode:**")
        use_atom_count = st.radio(
            "Choose execution method:",
            options=[False, True],
            format_func=lambda x: "Use supercell (-rc)" if not x else f"Specify atoms (-n {results['total_atoms']})",
            key="monitor_execution_mode"
        )

    with col_opt2:
        st.write("**Parallel Execution:**")
        enable_parallel = st.checkbox(
            "Enable parallel execution",
            value=False,
            help="Run multiple mcsqs instances simultaneously for faster convergence",
            key="monitor_enable_parallel"
        )

        if enable_parallel:
            parallel_runs = st.number_input(
                "Number of parallel runs:",
                min_value=2,
                max_value=100,
                value=3,
                step=1,
                key="monitor_parallel_count"
            )
        else:
            parallel_runs = 1

        st.write("**Time Limit:**")
        enable_time_limit = st.checkbox(
            "Set automatic time limit",
            value=False,
            help="Automatically stop mcsqs after specified time",
            key="monitor_enable_time_limit"
        )

        if enable_time_limit:
            time_limit_minutes = st.number_input(
                "Time limit (minutes):",
                min_value=1,
                max_value=10080,
                value=30,
                step=5,
                key="monitor_time_limit"
            )
        else:
            time_limit_minutes = None

    with col_opt3:
        st.write("**Cluster Settings:**")
        pair_cutoff = results.get('pair_cutoff', 1.1)
        triplet_cutoff = results.get('triplet_cutoff', None)
        quadruplet_cutoff = results.get('quadruplet_cutoff', None)

        st.write(f"Pair cutoff: {round(pair_cutoff, 3)}")
        if triplet_cutoff:
            st.write(f"Triplet cutoff: {triplet_cutoff}")
        if quadruplet_cutoff:
            st.write(f"Quadruplet cutoff: {quadruplet_cutoff}")

    if enable_parallel:
        cmd_preview = f"mcsqs {'-n ' + str(results['total_atoms']) if use_atom_count else '-rc'} # with {parallel_runs} parallel instances"
        log_info = "Will monitor mcsqs1.log for parallel execution"
    else:
        cmd_preview = f"mcsqs {'-n ' + str(results['total_atoms']) if use_atom_count else '-rc'} # single run"
        log_info = "Will monitor mcsqs.log for single execution"

    st.write("**Command Preview:**")
    st.code(cmd_preview, language="bash")
    st.caption(log_info)
    col_download, col_info = st.columns([1, 1])

    with col_download:
        if st.button("🛠️ Generate Monitor Script", type="tertiary", key="generate_monitor_script"):
            try:
                script_content = generate_atat_monitor_script(
                    results=results,
                    use_atom_count=use_atom_count,
                    parallel_runs=parallel_runs,
                    pair_cutoff=pair_cutoff,
                    triplet_cutoff=triplet_cutoff,
                    quadruplet_cutoff=quadruplet_cutoff,
                    max_param=results.get('max_param', 1.0),
                    time_limit_minutes=time_limit_minutes
                )

                st.download_button(
                    label="📥 Download monitor.sh",
                    data=script_content,
                    file_name="monitor.sh",
                    mime="text/plain",
                    type="primary",
                    key="download_monitor_script",
                )

                st.success("✅ Monitor script generated successfully!")
                with st.expander("Script Preview", expanded=False):
                    st.code(script_content, language="bash")
            except Exception as e:
                st.error(f"Error generating script: {str(e)}")

    with col_info:
        with st.expander("📖 How to use monitor.sh", expanded=False):
            st.markdown(f"""
            ### Usage Instructions:

            1. **Download the script** and place it in your ATAT working directory

            2. **Make it executable:**
               ```bash
               sudo chmod +x monitor.sh # or 'bash monitor.sh'
               ```

            3. **Run the script:**
               ```bash
               ./monitor.sh
               ```

            ### What the script does:
            - 🔧 **Auto-creates** `rndstr.in` and `sqscell.out` 
            - 🔧 **Runs corrdump** with your cluster settings
            - 🚀 **Starts mcsqs** in single instance (or in parallel if enabled)
            - 📊 **Monitors progress** every minute
            - 📁 **Saves additional monitored data** to `mcsqs_progress.csv`
            - 📁 **Converts bestsqs.out to POSCAR** automatically when stopped

            ### Output files:
            - **mcsqs_progress.csv** - Time-based progress data (upload this to analyze!)
            - **{"mcsqs1.log" if enable_parallel else "mcsqs.log"}** - MCSQS log file
            - **bestsqs.out** - Best SQS structure found
            - **bestcorr.out** - Correlation functions

            ### Configuration:
            - **Execution**: {cmd_preview}
            - **Monitoring**: Every 1 minute
            - **Stop the run**: Once user presses Ctrl+C

             **The generated CSV file can be uploaded back to this tool for analysis!**
            """)



def create_vacancies_from_sqs(sqs_structure, elements_to_remove):
    ordered_structure = prepare_structure_for_prdf(sqs_structure)

    original_composition = ordered_structure.composition
    original_total_atoms = len(ordered_structure)

    sites_to_keep = []
    sites_removed = []
    removal_counts = {}

    for i, site in enumerate(ordered_structure):
        element_symbol = site.specie.symbol

        if element_symbol not in elements_to_remove:
            sites_to_keep.append(i)
        else:
            sites_removed.append(i)
            if element_symbol not in removal_counts:
                removal_counts[element_symbol] = 0
            removal_counts[element_symbol] += 1

    if not sites_removed:
        raise ValueError(f"No atoms found for elements: {elements_to_remove}")

    if not sites_to_keep:
        raise ValueError("Cannot remove all atoms from the structure!")

    new_lattice = ordered_structure.lattice
    new_species = []
    new_coords = []

    for site_idx in sites_to_keep:
        site = ordered_structure[site_idx]
        new_species.append(site.specie)
        new_coords.append(site.frac_coords)

    vacancy_structure = Structure(
        lattice=new_lattice,
        species=new_species,
        coords=new_coords,
        coords_are_cartesian=False
    )

    total_removed = len(sites_removed)
    removal_percentage = (total_removed / original_total_atoms) * 100
    removal_info = {
        'original_composition': str(original_composition),
        'original_atom_count': original_total_atoms,
        'final_atom_count': len(vacancy_structure),
        'total_atoms_removed': total_removed,
        'removal_percentage': removal_percentage,
        'elements_removed': elements_to_remove,
        'removal_counts': removal_counts,
        'final_composition': str(vacancy_structure.composition),
        'density_change': f"{(1 - len(vacancy_structure) / original_total_atoms) * 100:.1f}% reduction"
    }

    return vacancy_structure, removal_info


def render_vacancy_creation_section(sqs_pymatgen_structure):
    st.markdown("---")
    st.subheader("🕳️ Create Vacancies in SQS Structure")
    st.info("""
    **Remove specific elements** from your SQS structure to create ordered vacancies.
    This is useful for studying vacancy formation, ion diffusion, or defect structures.
    """)

    structure_elements = []
    element_counts = {}

    for site in sqs_pymatgen_structure:
        if site.is_ordered:
            element = site.specie.symbol
            if element not in structure_elements:
                structure_elements.append(element)
            element_counts[element] = element_counts.get(element, 0) + 1
        else:
            for sp in site.species:
                element = sp.symbol
                if element not in structure_elements:
                    structure_elements.append(element)
                element_counts[element] = element_counts.get(element, 0) + site.species[sp]

    structure_elements.sort()
    st.write("**Current Structure Composition:**")
    comp_cols = st.columns(min(len(structure_elements), 4))
    for i, element in enumerate(structure_elements):
        with comp_cols[i % len(comp_cols)]:
            count = int(element_counts[element])
            st.metric(f"{element}", f"{count} atoms")

    st.write(f"**Total atoms:** {len(sqs_pymatgen_structure)}")

    col_select, col_preview = st.columns([1, 1])

    with col_select:
        st.write("**Select elements to remove:**")
        elements_to_remove = st.multiselect(
            "Choose elements to create vacancies:",
            options=structure_elements,
            default=[],
            help="Select one or more elements to remove from the structure",
            key="vacancy_elements_selector"
        )

        if elements_to_remove:
            st.write("**Elements to remove:**")
            for element in elements_to_remove:
                count = int(element_counts[element])
                percentage = (count / len(sqs_pymatgen_structure)) * 100
                st.write(f"- **{element}:** {count} atoms ({percentage:.1f}%)")

            total_to_remove = sum(int(element_counts[element]) for element in elements_to_remove)
            remaining_atoms = len(sqs_pymatgen_structure) - total_to_remove
            st.write(f"**Total removal:** {total_to_remove} atoms")
            st.write(f"**Remaining atoms:** {remaining_atoms}")

            if remaining_atoms == 0:
                st.error("⚠️ Cannot remove all atoms from the structure!")
            elif remaining_atoms < 10:
                st.warning("⚠️ Very few atoms will remain. Consider removing fewer elements.")

    with col_preview:
        if elements_to_remove and len(elements_to_remove) > 0:
            st.write("**Vacancy Creation Preview:**")
            total_to_remove = sum(int(element_counts[element]) for element in elements_to_remove)
            remaining_atoms = len(sqs_pymatgen_structure) - total_to_remove
            vacancy_percentage = (total_to_remove / len(sqs_pymatgen_structure)) * 100

            preview_data = []
            preview_data.append({"Property": "Original atoms", "Value": len(sqs_pymatgen_structure)})
            preview_data.append({"Property": "Atoms to remove", "Value": total_to_remove})
            preview_data.append({"Property": "Final atoms", "Value": remaining_atoms})
            preview_data.append({"Property": "Vacancy %", "Value": f"{vacancy_percentage:.1f}%"})

            preview_df = pd.DataFrame(preview_data)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

    create_vacancies_btn = st.button(
        "🕳️ Create Vacancy Structure",
        disabled=not elements_to_remove or remaining_atoms <= 0,
        type="primary",
        key="create_vacancies_btn"
    )

    if create_vacancies_btn and elements_to_remove:
        try:
            with st.spinner("Creating vacancy structure..."):
                vacancy_structure, removal_info = create_vacancies_from_sqs(
                    sqs_pymatgen_structure, elements_to_remove
                )

            st.success("✅ Vacancy structure created successfully!")
            st.subheader("📊 Vacancy Creation Results")

            col_info1, col_info2, col_info3 = st.columns(3)

            with col_info1:
                st.metric("Original Atoms", removal_info['original_atom_count'])
                st.metric("Final Atoms", removal_info['final_atom_count'])

            with col_info2:
                st.metric("Atoms Removed", removal_info['total_atoms_removed'])
                st.metric("Removal %", f"{removal_info['removal_percentage']:.1f}%")

            with col_info3:
                st.metric("Density Change", removal_info['density_change'])
                st.write(f"**Elements removed:** {', '.join(elements_to_remove)}")

            st.write("**Detailed Removal Information:**")
            removal_data = []
            for element, count in removal_info['removal_counts'].items():
                original_count = int(element_counts[element])
                removal_data.append({
                    "Element": element,
                    "Original Count": original_count,
                    "Atoms Removed": count,
                    "Removal %": f"{(count / original_count) * 100:.1f}%"
                })

            removal_df = pd.DataFrame(removal_data)
            st.dataframe(removal_df, use_container_width=True, hide_index=True)

            st.write("**Composition Comparison:**")
            comp_comparison = pd.DataFrame({
                "Property": ["Original", "With Vacancies"],
                "Composition": [removal_info['original_composition'], removal_info['final_composition']],
                "Total Atoms": [removal_info['original_atom_count'], removal_info['final_atom_count']]
            })
            st.dataframe(comp_comparison, use_container_width=True, hide_index=True)

            st.subheader("🔍 Vacancy Structure Visualization")
            sqs_result_with_vacancies = {'structure': vacancy_structure}
            sqs_visualization(sqs_result_with_vacancies)

            st.subheader("📥 Download Vacancy Structure")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                vacancy_poscar = generate_poscar_from_structure(
                    vacancy_structure,
                    f"Vacancy_SQS_{'-'.join(elements_to_remove)}_removed"
                )

                st.download_button(
                    label="📥 POSCAR (VASP)",
                    data=vacancy_poscar,
                    file_name=f"POSCAR_vacancy_{'-'.join(elements_to_remove)}.vasp",
                    mime="text/plain",
                    key="download_vacancy_poscar"
                )

            with col_dl2:
                try:
                    vacancy_cif, _ = generate_additional_format(
                        vacancy_structure, "CIF", f"vacancy_{'-'.join(elements_to_remove)}"
                    )

                    st.download_button(
                        label="📥 CIF Format",
                        data=vacancy_cif,
                        file_name=f"vacancy_{'-'.join(elements_to_remove)}.cif",
                        mime="chemical/x-cif",
                        key="download_vacancy_cif"
                    )
                except Exception as e:
                    st.button(
                        "📥 CIF Format",
                        disabled=True,
                        help=f"CIF generation failed: {str(e)}"
                    )

            with col_dl3:
                summary_report = generate_vacancy_summary_report(removal_info, elements_to_remove)

                st.download_button(
                    label="📥 Summary Report",
                    data=summary_report,
                    file_name=f"vacancy_report_{'-'.join(elements_to_remove)}.txt",
                    mime="text/plain",
                    key="download_vacancy_report"
                )
            if 'vacancy_structures' not in st.session_state:
                st.session_state.vacancy_structures = {}

            vacancy_key = f"vacancy_{'-'.join(elements_to_remove)}"
            st.session_state.vacancy_structures[vacancy_key] = {
                'structure': vacancy_structure,
                'removal_info': removal_info,
                'elements_removed': elements_to_remove
            }

        except Exception as e:
            st.error(f"Error creating vacancy structure: {str(e)}")
            import traceback
            st.error(f"Debug info: {traceback.format_exc()}")


def generate_poscar_from_structure(structure, comment="Generated Structure"):
    from pymatgen.io.vasp import Poscar

    ordered_structure = prepare_structure_for_prdf(structure)

    poscar = Poscar(ordered_structure, comment=comment)
    return str(poscar)


def generate_vacancy_summary_report(removal_info, elements_removed):
    report_lines = [
        "VACANCY STRUCTURE CREATION REPORT",
        "=" * 40,
        "",
        f"Elements Removed: {', '.join(elements_removed)}",
        f"Original Composition: {removal_info['original_composition']}",
        f"Final Composition: {removal_info['final_composition']}",
        "",
        "ATOM COUNT SUMMARY:",
        f"  Original atoms: {removal_info['original_atom_count']}",
        f"  Final atoms: {removal_info['final_atom_count']}",
        f"  Total removed: {removal_info['total_atoms_removed']}",
        f"  Removal percentage: {removal_info['removal_percentage']:.2f}%",
        "",
        "DETAILED REMOVAL COUNTS:",
    ]

    for element, count in removal_info['removal_counts'].items():
        report_lines.append(f"  {element}: {count} atoms removed")

    report_lines.extend([
        "",
        "STRUCTURE PROPERTIES:",
        f"  Density change: {removal_info['density_change']}",
        f"  Lattice: Unchanged (original lattice preserved)",
        f"  Symmetry: May be reduced due to vacancy formation",
        "",
        "NOTES:",
        "- Vacancies created by completely removing selected atoms",
        "- Lattice parameters remain unchanged",
        "- Structure can be used for defect studies, diffusion analysis",
        "- Consider relaxing the structure with DFT calculations",
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ])

    return "\n".join(report_lines)


# Correlation

def render_correlation_analysis_tab():
    st.write("**Upload bestcorr.out file to analyze correlation functions:**")
    uploaded_bestcorr = st.file_uploader(
        "Upload bestcorr.out file:",
        type=['out', 'txt'],
        help="Upload the bestcorr.out file generated by ATAT mcsqs optimization",
        key="bestcorr_uploader"
    )

    if uploaded_bestcorr is not None:
        try:
            file_content = uploaded_bestcorr.read().decode('utf-8')
            correlation_data, objective_function = parse_bestcorr_file(file_content)

            if not correlation_data:
                st.warning("No correlation data found in the file.")
                st.info("Please ensure the file contains correlation function data.")
                return

            st.success(f"✅ Successfully parsed {len(correlation_data)} correlation functions!")

            if objective_function is not None:
                st.metric("Objective Function", f"{objective_function:.6f}")

            st.subheader("🎯 SQS Quality Assessment")

            ordering_analysis = analyze_ordering_from_correlations(correlation_data)

            col_status1, col_status2 = st.columns([1, 1])

            with col_status1:
                status = ordering_analysis['overall_status']
                recommendation = ordering_analysis['recommendation']

                if status == "Excellent SQS":
                    st.success(f"🎯 **{status}**")
                    st.success(f"✅ **{recommendation}**")
                elif status == "Good SQS":
                    st.success(f"✅ **{status}**")
                    st.info(f"ℹ️ **{recommendation}**")
                elif status == "Fair SQS":
                    st.warning(f"⚠️ **{status}**")
                    st.warning(f"⚠️ **{recommendation}**")
                else:
                    st.error(f"❌ **{status}**")
                    st.error(f"🔄 **{recommendation}**")

            with col_status2:
                st.write("**Detailed Assessment:**")
                st.write(f"• **Randomness Score:** {ordering_analysis['randomness_score']:.3f}/1.0")
                st.write(f"• **Max Deviation:** {ordering_analysis['max_abs_difference']:.4f}")
                st.write(f"• **Average Deviation:** {ordering_analysis['avg_abs_difference']:.4f}")
                if ordering_analysis['perfect_matches'] > 0:
                    st.write(f"• **Perfect Matches:** {ordering_analysis['perfect_matches']}")

            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

            with col_metrics1:
                st.metric("Strong Deviations", ordering_analysis['strong_deviations'])
                st.metric("Moderate Deviations", ordering_analysis['moderate_deviations'])

            with col_metrics2:
                st.metric("Weak Deviations", ordering_analysis['weak_deviations'])
                st.metric("Perfect Matches", ordering_analysis['perfect_matches'])

            with col_metrics3:
                st.metric("Total Correlations", ordering_analysis['total_correlations'])
                st.metric("Max Distance", f"{ordering_analysis['max_distance']:.3f} Å")

            st.subheader("📈 Correlation Analysis")
            correlation_plot = create_correlation_distance_plot(correlation_data)
            st.plotly_chart(correlation_plot, use_container_width=True)

            col_interpretation1, col_interpretation2 = st.columns(2)

            with col_interpretation1:
                st.subheader("🔍 Interpretation Guide")
                st.write("**Understanding Correlations:**")
                st.write("• **Target**: Expected value for perfect random alloy")
                st.write("• **SQS**: Actual value in your structure")
                st.write("• **Difference**: How far you are from ideal randomness")
                st.write("")
                st.write("**Deviation Categories:**")
                st.write("• **Strong (>0.1)**: Significant ordering/clustering")
                st.write("• **Moderate (0.05-0.1)**: Some non-random behavior")
                st.write("• **Weak (<0.05)**: Nearly random (good!)")
                st.write("• **Perfect (=0.0)**: Exact match to target")

            with col_interpretation2:
                st.subheader("📋 Distance Analysis")
                distance_analysis = analyze_correlations_by_distance(correlation_data)

                for distance, data in list(distance_analysis.items())[:5]:
                    with st.expander(f"Distance {distance:.3f} Å ({data['count']} correlations)", expanded=False):
                        st.write(f"**Average SQS correlation:** {data['avg_sqs_correlation']:.4f}")
                        st.write(f"**Max |difference|:** {data['max_abs_difference']:.4f}")
                        st.write(f"**Standard deviation:** {data['std_difference']:.4f}")

                        if data['max_abs_difference'] > 0.1:
                            st.error("❌ Significant deviations at this distance")
                        elif data['max_abs_difference'] > 0.05:
                            st.warning("⚠️ Moderate deviations at this distance")
                        else:
                            st.success("✅ Good randomness at this distance")

            st.subheader("📊 Detailed Correlation Data")

            correlation_df = pd.DataFrame(correlation_data)
            correlation_df['Abs_Target_Correlation'] = correlation_df['Target_Correlation'].abs()
            correlation_df['Abs_SQS_Correlation'] = correlation_df['SQS_Correlation'].abs()
            correlation_df['Difference'] = correlation_df['SQS_Correlation'] - correlation_df['Target_Correlation']
            correlation_df['Abs_Difference'] = correlation_df['Difference'].abs()

            correlation_df['Quality'] = correlation_df['Abs_Difference'].apply(
                lambda x: "Perfect" if x < 0.001 else "Weak" if x <= 0.05 else "Moderate" if x <= 0.1 else "Strong"
            )

            correlation_df = correlation_df.round(6)

            st.dataframe(correlation_df, use_container_width=True)

            col_stats1, col_stats2 = st.columns(2)

            with col_stats1:
                st.write("**Statistical Summary:**")
                st.write(f"• **Total correlations:** {len(correlation_data)}")
                st.write(f"• **Average |SQS correlation|:** {correlation_df['Abs_SQS_Correlation'].mean():.4f}")
                st.write(f"• **Average |difference|:** {correlation_df['Abs_Difference'].mean():.4f}")
                st.write(f"• **Max |difference|:** {correlation_df['Abs_Difference'].max():.4f}")

            with col_stats2:
                strong_count = len(correlation_df[correlation_df['Abs_Difference'] > 0.1])
                moderate_count = len(correlation_df[(correlation_df['Abs_Difference'] > 0.05) &
                                                    (correlation_df['Abs_Difference'] <= 0.1)])
                weak_count = len(correlation_df[(correlation_df['Abs_Difference'] > 0.001) &
                                                (correlation_df['Abs_Difference'] <= 0.05)])
                perfect_count = len(correlation_df[correlation_df['Abs_Difference'] <= 0.001])

                st.write("**Deviation Distribution:**")
                st.write(f"• **Strong deviations (>0.1):** {strong_count}")
                st.write(f"• **Moderate deviations (0.05-0.1):** {moderate_count}")
                st.write(f"• **Weak deviations (<0.05):** {weak_count}")
                st.write(f"• **Perfect matches (<0.001):** {perfect_count}")

            csv_data = correlation_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Correlation Analysis (CSV)",
                data=csv_data,
                file_name="correlation_analysis.csv",
                mime="text/csv",
                key="download_correlation_csv"
            )

        except UnicodeDecodeError:
            st.error("Error reading bestcorr.out file. Please ensure the file is a text file with UTF-8 encoding.")
        except Exception as e:
            st.error(f"Error processing bestcorr.out file: {str(e)}")
            import traceback
            st.error(f"Debug info: {traceback.format_exc()}")


def parse_bestcorr_file(file_content):
    lines = file_content.strip().split('\n')
    correlation_data = []
    objective_function = None

    for line in lines:
        line = line.strip()
        if line.startswith('Objective_function='):
            objective_function = float(line.split('=')[1].strip())
        elif line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cluster_type = int(parts[0])
                    distance = float(parts[1])
                    sqs_correlation = float(parts[2])
                    target_correlation = float(parts[3])

                    correlation_data.append({
                        'Cluster_Type': cluster_type,
                        'Distance': distance,
                        'SQS_Correlation': sqs_correlation,
                        'Target_Correlation': target_correlation
                    })
                except (ValueError, IndexError):
                    continue

    return correlation_data, objective_function


def analyze_ordering_from_correlations(correlation_data):
    if not correlation_data:
        return {}

    differences = [abs(data['SQS_Correlation'] - data['Target_Correlation']) for data in correlation_data]
    sqs_correlations = [abs(data['SQS_Correlation']) for data in correlation_data]
    distances = [data['Distance'] for data in correlation_data]

    max_abs_difference = max(differences)
    avg_abs_difference = sum(differences) / len(differences)

    strong_deviations = sum(1 for diff in differences if diff > 0.1)
    moderate_deviations = sum(1 for diff in differences if 0.05 < diff <= 0.1)
    weak_deviations = sum(1 for diff in differences if 0.001 < diff <= 0.05)
    perfect_matches = sum(1 for diff in differences if diff <= 0.001)

    randomness_score = 1.0 - (avg_abs_difference * 2)
    randomness_score = max(0, min(1, randomness_score))

    if perfect_matches >= len(correlation_data) * 0.8 and max_abs_difference <= 0.05:
        overall_status = "Excellent SQS"
        recommendation = "Structure is highly disordered and ideal for calculations"
    elif strong_deviations == 0 and max_abs_difference <= 0.1:
        overall_status = "Good SQS"
        recommendation = "Structure is well-disordered and should be suitable for most calculations"
    elif strong_deviations <= len(correlation_data) * 0.2:
        overall_status = "Fair SQS"
        recommendation = "Structure has some ordering - consider longer mcsqs run"
    else:
        overall_status = "Poor SQS"
        recommendation = "Structure shows significant ordering - extend mcsqs optimization"

    return {
        'overall_status': overall_status,
        'recommendation': recommendation,
        'randomness_score': randomness_score,
        'max_abs_difference': max_abs_difference,
        'avg_abs_difference': avg_abs_difference,
        'strong_deviations': strong_deviations,
        'moderate_deviations': moderate_deviations,
        'weak_deviations': weak_deviations,
        'perfect_matches': perfect_matches,
        'max_distance': max(distances) if distances else 0,
        'total_correlations': len(correlation_data)
    }


def analyze_correlations_by_distance(correlation_data):
    distance_groups = {}

    for data in correlation_data:
        distance = round(data['Distance'], 3)
        if distance not in distance_groups:
            distance_groups[distance] = []
        distance_groups[distance].append({
            'sqs_correlation': data['SQS_Correlation'],
            'target_correlation': data['Target_Correlation'],
            'difference': data['SQS_Correlation'] - data['Target_Correlation']
        })

    distance_analysis = {}
    for distance, correlations in distance_groups.items():
        sqs_values = [corr['sqs_correlation'] for corr in correlations]
        differences = [abs(corr['difference']) for corr in correlations]

        distance_analysis[distance] = {
            'count': len(correlations),
            'avg_sqs_correlation': sum(sqs_values) / len(sqs_values),
            'max_abs_difference': max(differences),
            'std_difference': (sum((d - sum(differences) / len(differences)) ** 2 for d in differences) / len(
                differences)) ** 0.5
        }

    return dict(sorted(distance_analysis.items()))


def create_correlation_distance_plot(correlation_data):
    import plotly.graph_objects as go

    distances = [data['Distance'] for data in correlation_data]
    sqs_correlations = [data['SQS_Correlation'] for data in correlation_data]
    target_correlations = [data['Target_Correlation'] for data in correlation_data]
    differences = [abs(data['SQS_Correlation'] - data['Target_Correlation']) for data in correlation_data]
    cluster_types = [data['Cluster_Type'] for data in correlation_data]

    colors = ['red' if diff > 0.1 else 'orange' if diff > 0.05 else 'green' for diff in differences]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=distances,
        y=target_correlations,
        mode='markers',
        name='Target (Random Alloy)',
        marker=dict(
            color='blue',
            size=8,
            symbol='circle',
            opacity=0.7
        ),
        hovertemplate='<b>Target (Random)</b><br>Distance: %{x:.3f} Å<br>Correlation: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=distances,
        y=sqs_correlations,
        mode='markers',
        name='SQS Structure',
        marker=dict(
            color=colors,
            size=10,
            symbol='diamond',
            opacity=0.8
        ),
        hovertemplate='<b>SQS Structure</b><br>Distance: %{x:.3f} Å<br>Correlation: %{y:.4f}<br>Cluster: %{customdata}-body<extra></extra>',
        customdata=cluster_types
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.1, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="Strong deviation threshold (+0.1)")
    fig.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="Strong deviation threshold (-0.1)")
    fig.add_hline(y=0.05, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_hline(y=-0.05, line_dash="dot", line_color="orange", opacity=0.3)

    fig.update_layout(
        title=dict(
            text="SQS vs Target Correlation Functions",
            font=dict(size=24, family="Arial Black")
        ),
        xaxis_title="Inter-atomic Distance (Å)",
        yaxis_title="Correlation Function Value",
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=14)
        ),
        font=dict(size=16, family="Arial"),
        xaxis=dict(
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title_font=dict(size=18, family="Arial Black"),
            tickfont=dict(size=14)
        )
    )

    return fig
