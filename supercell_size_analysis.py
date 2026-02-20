import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import io
import zipfile
from collections import defaultdict, Counter


def analyze_supercell_size_range(working_structure, target_concentrations, use_sublattice_mode,
                                 chem_symbols, base_transformation_matrix, size_range,
                                 n_samples_per_size, GeometryCacheUI, perform_live_analysis_rigorous,
                                 parse_rndstr_content, generate_atat_rndstr_content_corrected):
    results_by_size = []

    status_container = st.empty()
    progress_bar = st.progress(0)
    total_steps = len(size_range)

    base_nx = int(base_transformation_matrix[0, 0])
    base_ny = int(base_transformation_matrix[1, 1])
    base_nz = int(base_transformation_matrix[2, 2])

    for step_idx, increment in enumerate(size_range):
        nx = base_nx + increment
        ny = base_ny + increment
        nz = base_nz + increment

        status_container.info(f"🔄 Analyzing supercell size {nx}×{ny}×{nz}...")

        transformation_matrix = base_transformation_matrix.copy()
        transformation_matrix[0, 0] = nx
        transformation_matrix[1, 1] = ny
        transformation_matrix[2, 2] = nz

        total_atoms = len(working_structure) * nx * ny * nz

        try:
            rndstr_content = generate_atat_rndstr_content_corrected(
                working_structure, target_concentrations, use_sublattice_mode,
                chem_symbols, transformation_matrix
            )
            sites = parse_rndstr_content(rndstr_content)

            cache = GeometryCacheUI(working_structure.lattice, sites, nx, ny, nz)

            scores_for_this_size = []
            rmse_for_this_size = []
            structures_for_this_size = []

            concentration_check = None

            sublattice_scores_accumulated = defaultdict(list)

            for sample_idx in range(n_samples_per_size):
                base_seed = int(time.time()) % (2 ** 31)
                seed = (base_seed + sample_idx + step_idx * 1000) % (2 ** 32 - 1)

                elements = cache.generate_random_config(seed)
                score, rmse, _, sub_scores = perform_live_analysis_rigorous(cache, elements)

                scores_for_this_size.append(score)
                rmse_for_this_size.append(rmse)
                structures_for_this_size.append({
                    'elements': elements.copy(),
                    'seed': seed,
                    'score': score,
                    'sample_id': sample_idx + 1
                })

                for k, v in sub_scores.items():
                    sublattice_scores_accumulated[k].append(v)

                if sample_idx == 0:
                    element_counts = Counter(elements)
                    total = len(elements)
                    actual_concs = {el: count / total for el, count in element_counts.items()}
                    concentration_check = {
                        'actual_counts': dict(element_counts),
                        'actual_concentrations': actual_concs,
                        'total_atoms': total
                    }

            mean_score = np.mean(scores_for_this_size)
            std_score = np.std(scores_for_this_size)
            mean_rmse = np.mean(rmse_for_this_size)
            std_rmse = np.std(rmse_for_this_size)

            results_by_size.append({
                'increment': increment,
                'supercell_dims': f"{nx}×{ny}×{nz}",
                'total_atoms': total_atoms,
                'mean_score': mean_score,
                'std_score': std_score,
                'mean_rmse': mean_rmse,
                'std_rmse': std_rmse,
                'individual_scores': scores_for_this_size,
                'individual_rmse': rmse_for_this_size,
                'concentration_check': concentration_check,
                'structures': structures_for_this_size,
                'cache': cache,
                'nx': nx,
                'ny': ny,
                'nz': nz,
                'per_sublattice_mean': {k: np.mean(v) for k, v in sublattice_scores_accumulated.items()},
                'per_sublattice_std':  {k: np.std(v)  for k, v in sublattice_scores_accumulated.items()},
            })

        except Exception as e:
            st.warning(f"Failed to analyze size {size_multiplier}×{size_multiplier}×{size_multiplier}: {e}")
            continue

        progress_bar.progress((step_idx + 1) / total_steps)

    progress_bar.empty()
    status_container.empty()

    return results_by_size


def create_size_analysis_plots(results_by_size, selected_sublattice="Overall"):
    if not results_by_size:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Mean Randomness Score vs Supercell Size"
            + ("" if selected_sublattice == "Overall" else f" — {selected_sublattice}"),
            "Computational Cost (N atoms)"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        horizontal_spacing=0.12
    )
    fig.update_annotations(font_size=20)

    dims_labels = [r['supercell_dims'] for r in results_by_size]
    dims = [r['supercell_dims'] for r in results_by_size]
    n_atoms = [r['total_atoms'] for r in results_by_size]

    if selected_sublattice == "Overall":
        mean_scores = [r['mean_score'] for r in results_by_size]
        std_scores  = [r['std_score']  for r in results_by_size]
        trace_name  = "Mean Score (Overall)"
        trace_color = '#2E86C1'
    else:
        mean_scores = [r['per_sublattice_mean'].get(selected_sublattice, float('nan')) for r in results_by_size]
        std_scores  = [r['per_sublattice_std'].get(selected_sublattice, 0.0)           for r in results_by_size]
        trace_name  = f"Mean Score ({selected_sublattice})"
        trace_color = '#C0392B'

    fig.add_trace(
        go.Scatter(
            x=dims_labels,
            y=mean_scores,
            mode='lines+markers',
            name=trace_name,
            line=dict(color=trace_color, width=2),
            marker=dict(size=10),
            error_y=dict(type='data', array=std_scores, visible=True, color=trace_color, thickness=1.5),
            customdata=list(zip(dims, n_atoms)),
            hovertemplate='<b style="font-size:22px">Supercell: %{x}</b><br>' +
                          '<b style="font-size:22px">Score: %{y:.4f}</b><br>' +
                          '<b style="font-size:22px">N atoms: %{customdata[1]}</b><br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                  annotation_text="Great", row=1, col=1)
    fig.add_hline(y=0.85, line_dash="dash", line_color="orange",
                  annotation_text="Good", row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=dims_labels,
            y=n_atoms,
            name='Total Atoms',
            marker=dict(color='#3498DB'),
            customdata=dims,
            hovertemplate='<b style="font-size:22px">Supercell: %{x}</b><br>' +
                          '<b style="font-size:22px">N atoms: %{y}</b><br>' +
                          '<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Supercell Size", title_font=dict(size=24),
                     tickfont=dict(size=20), row=1, col=1)
    fig.update_yaxes(title_text="Randomness Score", title_font=dict(size=24),
                     tickfont=dict(size=20), row=1, col=1)

    fig.update_xaxes(title_text="Supercell Size", title_font=dict(size=24),
                     tickfont=dict(size=20), row=1, col=2)
    fig.update_yaxes(title_text="Number of Atoms", title_font=dict(size=24),
                     tickfont=dict(size=20), row=1, col=2)

    fig.update_layout(
        height=600,
        font=dict(family="Arial", size=20, color="black"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=20)),
        plot_bgcolor='rgba(245,245,245,0.5)',
        margin=dict(l=80, r=140, t=80, b=100),
        hoverlabel=dict(
            bgcolor="white",
            font_size=22,
            font_family="Arial",
            bordercolor="black"
        )
    )

    fig.update_xaxes(
        spikemode='across',
        spikethickness=1,
        spikecolor='gray',
        row=1, col=1
    )
    fig.update_yaxes(
        spikemode='across',
        spikethickness=1,
        spikecolor='gray',
        row=1, col=1
    )

    fig.update_xaxes(
        spikemode='across',
        spikethickness=1,
        spikecolor='gray',
        row=1, col=2
    )
    fig.update_yaxes(
        spikemode='across',
        spikethickness=1,
        spikecolor='gray',
        row=1, col=2
    )

    return fig


def create_vasp_content(lattice_matrix, atoms_frac, elements, comment="Random Structure"):
    from collections import defaultdict
    grouped = defaultdict(list)
    for frac_pos, elem in zip(atoms_frac, elements):
        grouped[elem].append(frac_pos)

    sorted_elements = sorted(grouped.keys())

    lines = [comment, "1.0"]
    for vec in lattice_matrix:
        lines.append(f" {vec[0]:15.9f} {vec[1]:15.9f} {vec[2]:15.9f}")

    lines.append(" ".join(sorted_elements))
    lines.append(" ".join([str(len(grouped[el])) for el in sorted_elements]))
    lines.append("Direct")

    for el in sorted_elements:
        for pos in grouped[el]:
            p = pos % 1.0
            lines.append(f" {p[0]:15.9f} {p[1]:15.9f} {p[2]:15.9f}")

    return "\n".join(lines)


def create_structures_zip(results_by_size):
    import io
    import zipfile

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for result in results_by_size:
            supercell_dims = result['supercell_dims']
            folder_name = f"supercell_{supercell_dims.replace('×', 'x')}"

            cache = result['cache']
            structures = result['structures']

            for struct_info in structures:
                elements = struct_info['elements']
                score = struct_info['score']
                sample_id = struct_info['sample_id']
                seed = struct_info['seed']

                comment = f"{supercell_dims} Sample_{sample_id:03d} Score_{score:.4f} Seed_{seed}"

                vasp_content = create_vasp_content(
                    cache.lattice_matrix,
                    cache.atoms_frac,
                    elements,
                    comment
                )

                filename = f"{folder_name}/POSCAR_{sample_id:03d}_score_{score:.4f}.vasp"
                zf.writestr(filename, vasp_content)

        info_lines = [
            "Supercell Size Analysis - Structure Files",
            "=" * 60,
            "",
            "Directory Structure:",
            ""
        ]

        for result in results_by_size:
            supercell_dims = result['supercell_dims']
            n_structures = len(result['structures'])
            mean_score = result['mean_score']
            total_atoms = result['total_atoms']

            info_lines.append(f"supercell_{supercell_dims.replace('×', 'x')}/")
            info_lines.append(f"  - {n_structures} structures")
            info_lines.append(f"  - {total_atoms} atoms per structure")
            info_lines.append(f"  - Mean score: {mean_score:.4f}")
            info_lines.append("")

        info_lines.extend([
            "",
            "File Naming Convention:",
            "POSCAR_XXX_score_Y.YYYY.vasp",
            "  - XXX: Sample ID (001, 002, ...)",
            "  - Y.YYYY: Randomness score",
            "",
            "Total Files: " + str(sum(len(r['structures']) for r in results_by_size)),
            ""
        ])

        zf.writestr("README.txt", "\n".join(info_lines))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def render_supercell_size_analysis(working_structure, target_concentrations, transformation_matrix,
                                   use_sublattice_mode, chem_symbols, total_atoms,
                                   GeometryCacheUI, perform_live_analysis_rigorous,
                                   parse_rndstr_content, generate_atat_rndstr_content_corrected):
    with st.expander("ℹ️ What is this analysis?", expanded=False):
        st.markdown("""
        ### Purpose
        Analyze how randomness quality and computational cost scale with supercell size.

        ### How it works
        Starting from your **current supercell**, the analysis tests progressively larger sizes by 
        adding +1, +2, +3, etc. in each direction.

        **Example:** Current 2×2×2 with 3 increments
        - Tests: 2×2×2, 3×3×3, 4×4×4, 5×5×5

        ### Why this matters
        - **Larger supercells** → Better statistical sampling, lower variance, more random-alloy-like behavior
        - **Larger supercells** → Higher computational cost (more atoms = slower DFT/MLIP calculations)
        - **Goal**: Find the optimal balance between quality and cost

        ### What is analyzed
        For each supercell size (current, current+1, current+2, ...):
        1. Generate multiple random structures at that size
        2. Calculate Warren-Cowley parameters and randomness score
        3. Compute mean score and standard deviation
        4. Track total number of atoms (computational cost proxy)

        ### Interpretation
        - **Mean Score**: Average randomness quality at this size
        - **Std Dev**: Statistical fluctuations (should decrease with size)
        - **N atoms**: Computational cost indicator
        - **Optimal size**: Where score plateaus with acceptable variance
        """)

    base_nx = int(transformation_matrix[0, 0])
    base_ny = int(transformation_matrix[1, 1])
    base_nz = int(transformation_matrix[2, 2])

    col1, col2, col3 = st.columns(3)

    with col1:
        max_increments = st.number_input(
            "Number of size increments:",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            help="How many times to increase supercell by +1 in each direction"
        )

    with col2:
        n_samples = st.number_input(
            "Samples per size:",
            min_value=3,
            max_value=100,
            value=10,
            step=1,
            help="Number of random structures to generate at each size"
        )

    with col3:
        atom_limit = st.number_input(
            "Max atoms limit:",
            min_value=500,
            max_value=10000,
            value=5000,
            step=500,
            help="Skip supercells larger than this"
        )

    tested_sizes = []
    atom_counts = []
    exceed_limit = False
    max_safe_increment = max_increments

    for increment in range(0, max_increments + 1):
        nx = base_nx + increment
        ny = base_ny + increment
        nz = base_nz + increment
        n_atoms_in_cell = len(working_structure)
        n_atoms = n_atoms_in_cell * nx * ny * nz

        tested_sizes.append(f"{nx}×{ny}×{nz}")
        atom_counts.append(n_atoms)

        if n_atoms > atom_limit:
            if not exceed_limit:
                exceed_limit = True
                max_safe_increment = increment - 1

    st.info(f"**Current supercell**: {base_nx}×{base_ny}×{base_nz} ({total_atoms} atoms)")

    if exceed_limit and max_safe_increment < 0:
        st.error(
            f"❌ **Cannot run analysis**: The current supercell already has {atom_counts[0]} atoms, exceeding the limit of {atom_limit} atoms.")
        st.info("💡 **Solution**: Increase the atom limit or use a smaller base supercell.")
        return

    if exceed_limit:
        safe_sizes = tested_sizes[:max_safe_increment + 1]
        safe_atoms = atom_counts[:max_safe_increment + 1]
        exceeded_sizes = tested_sizes[max_safe_increment + 1:]
        exceeded_atoms = atom_counts[max_safe_increment + 1:]

        st.warning(
            f"⚠️ **Atom limit exceeded**: Sizes beyond +{max_safe_increment} would exceed {atom_limit} atoms. Analysis will be limited to +{max_safe_increment}.")

        size_info = " | ".join([f"{s} ({a} atoms)" for s, a in zip(safe_sizes, safe_atoms)])
        st.info(f"**Will test sizes**: {size_info}")

        exceeded_info = " | ".join([f"{s} ({a} atoms)" for s, a in zip(exceeded_sizes, exceeded_atoms)])
        st.caption(f"🚫 **Skipped (too large)**: {exceeded_info}")

        size_range = list(range(0, max_safe_increment + 1))
    else:
        size_info = " | ".join([f"{s} ({a} atoms)" for s, a in zip(tested_sizes, atom_counts)])
        st.info(f"**Will test sizes**: {size_info}")
        size_range = list(range(0, max_increments + 1))

    can_run = len(size_range) > 0

    run_button = st.button(
        "🚀 Run Supercell Size Analysis",
        type="primary",
        use_container_width=True,
        disabled=not can_run
    )

    if run_button and can_run:
        with st.spinner("Running analysis across supercell sizes..."):
            results = analyze_supercell_size_range(
                working_structure, target_concentrations, use_sublattice_mode,
                chem_symbols, transformation_matrix, size_range, n_samples,
                GeometryCacheUI, perform_live_analysis_rigorous,
                parse_rndstr_content, generate_atat_rndstr_content_corrected
            )

            st.session_state['size_analysis_results'] = results

    if 'size_analysis_results' in st.session_state:
        results = st.session_state['size_analysis_results']


        with st.expander("🔍 Concentration Verification", expanded=False):

            conc_data = []
            for r in results:
                if r.get('concentration_check'):
                    cc = r['concentration_check']
                    conc_str = ", ".join(
                        [f"{el}: {cc['actual_counts'][el]} ({cc['actual_concentrations'][el] * 100:.2f}%)"
                         for el in sorted(cc['actual_counts'].keys())])
                    conc_data.append({
                        'Size': r['supercell_dims'],
                        'Total Atoms': cc['total_atoms'],
                        'Composition': conc_str})
            if conc_data:
                conc_df = pd.DataFrame(conc_data)
                st.dataframe(conc_df, use_container_width=True, hide_index=True)
        #        st.caption(
        #            "💡 Concentrations are recalculated for each supercell size to match target as closely as possible with integer atom counts.")
        with st.expander("📊 Detailed Results Table", expanded=False):
            table_data = []
            for r in results:
                table_data.append({
                    'Supercell': r['supercell_dims'],
                    'Total Atoms': r['total_atoms'],
                    'Mean Score': f"{r['mean_score']:.4f}",
                    'Std Dev': f"{r['std_score']:.4f}",
                    'Mean RMSE': f"{r['mean_rmse']:.4f}",
                    'Samples': len(r['individual_scores'])
                })

            results_df = pd.DataFrame(table_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.caption(
                "💡 Mean Score: Average randomness quality | Std Dev: Statistical fluctuation | Lower RMSE = Better randomness")

        available_sublattices = []
        for r in results:
            for k in r.get('per_sublattice_mean', {}).keys():
                if k not in available_sublattices:
                    available_sublattices.append(k)

        if available_sublattices:
            selected_sublattice = st.radio(
                "📊 Plot score for:",
                options=["Overall"] + available_sublattices,
                index=0,
                horizontal=True,
                help="Switch between the aggregate score and an individual sublattice score"
            )
        else:
            selected_sublattice = "Overall"

        fig = create_size_analysis_plots(results, selected_sublattice=selected_sublattice)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📥 Download All Structures")

        total_structures = sum(len(r['structures']) for r in results)
        total_sizes = len(results)

        st.info(f"**Available:** {total_structures} structures across {total_sizes} supercell sizes")

        col_info, col_btn = st.columns([2, 1])

        with col_info:
            st.markdown("""
            **ZIP contents:**
            - Separate folder for each supercell size
            - All structures in VASP POSCAR format
            - Filename includes sample ID and score
            - README.txt with directory structure
            """)

        with col_btn:
            if st.button("🔽 Generate Download Package", type="primary", use_container_width=True):
                with st.spinner("Creating ZIP file..."):
                    zip_data = create_structures_zip(results)
                    st.session_state['structures_zip'] = zip_data
                st.success("✅ Package ready!")

        if 'structures_zip' in st.session_state:
            st.download_button(
                "💾 Download Structures ZIP",
                data=st.session_state['structures_zip'],
                file_name="supercell_size_analysis_structures.zip",
                mime="application/zip",
                use_container_width=True,
                type='primary'
            )
