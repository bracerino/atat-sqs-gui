import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import random
import time
from collections import defaultdict


def get_mic_distance_matrix(frac_coords, lattice_matrix):
    n = len(frac_coords)
    G = np.dot(lattice_matrix, lattice_matrix.T)
    rows, cols, dists = [], [], []

    for i in range(n):
        delta = frac_coords - frac_coords[i]
        delta -= np.round(delta)
        d2 = np.sum(np.dot(delta, G) * delta, axis=1)
        d = np.sqrt(d2)
        mask = (d > 1e-4) & (d < 8.0)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            rows.extend([i] * len(indices))
            cols.extend(indices)
            dists.extend(d[indices])

    return np.array(rows), np.array(cols), np.array(dists)


class GeometryCacheUI:

    def __init__(self, lattice, sites, nx, ny, nz):
        self.specs = sorted(list(set(s[3] for s in sites)))
        self.spec_map = {s: i for i, s in enumerate(self.specs)}

        atoms_frac, sub_ids = [], []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for x, y, z, spec in sites:
                        atoms_frac.append([(x + ix) / nx, (y + iy) / ny, (z + iz) / nz])
                        sub_ids.append(self.spec_map[spec])

        self.atoms_frac = np.array(atoms_frac)
        self.sub_ids = np.array(sub_ids)
        self.lattice_matrix = np.array([lattice.matrix[0] * nx, lattice.matrix[1] * ny, lattice.matrix[2] * nz])
        self.n_atoms = len(atoms_frac)

        self.sublattice_shells = {}
        self.debug_info = []

        self.G = np.dot(self.lattice_matrix, self.lattice_matrix.T)

        for sid, sp in enumerate(self.specs):
            if ',' not in sp: continue

            mask = (self.sub_ids == sid)
            indices = np.where(mask)[0]
            if len(indices) < 2: continue

            sub_frac = self.atoms_frac[indices]

            n_sub = len(sub_frac)
            dists_set = set()
            neighbor_map = defaultdict(lambda: defaultdict(list))

            for i in range(n_sub):
                d_vec = sub_frac - sub_frac[i]
                d_vec -= np.round(d_vec)
                d2 = np.sum(np.dot(d_vec, self.G) * d_vec, axis=1)
                d = np.sqrt(d2)

                d_rounded = np.round(d, 3)

                valid_mask = (d > 1e-4) & (d < 8.0)
                valid_indices = np.where(valid_mask)[0]

                for j_local in valid_indices:
                    dist_val = d_rounded[j_local]
                    dists_set.add(dist_val)
                    neighbor_map[dist_val][indices[i]].append(indices[j_local])

            shells = sorted(list(dists_set))[:3]

            shells_clean = [float(s) for s in shells]
            self.debug_info.append(f"Sublattice {sid + 1} [{sp}] ({len(indices)} sites) Shells: {shells_clean}")

            self.sublattice_shells[sid] = {'shells': shells, 'map': neighbor_map}

    def generate_random_config(self, seed):
        random.seed(seed);
        np.random.seed(seed)

        final_elements = [None] * len(self.sub_ids)

        sub_indices = defaultdict(list)
        for idx, sid in enumerate(self.sub_ids):
            sub_indices[sid].append(idx)

        for sid, indices in sub_indices.items():
            spec = self.specs[sid]
            n_sites = len(indices)

            if '=' in spec:
                atoms_for_sublattice = []
                parts = spec.split(',')
                els, fracs = [], []
                for part in parts:
                    e, f = part.split('=')
                    els.append(e)
                    fracs.append(float(f))

                counts = [int(round(f * n_sites)) for f in fracs]

                diff = n_sites - sum(counts)
                if diff != 0: counts[-1] += diff

                for el, count in zip(els, counts):
                    atoms_for_sublattice.extend([el] * count)

                random.shuffle(atoms_for_sublattice)

                for i, atom in zip(indices, atoms_for_sublattice):
                    final_elements[i] = atom
            else:
                for i in indices:
                    final_elements[i] = spec

        return final_elements


def perform_live_analysis_rigorous(cache, elements):
    total_rmse = 0.0
    sub_count = 0
    breakdown_list = []
    sublattice_scores = {}
    elements = np.array(elements)

    for sid, data in cache.sublattice_shells.items():
        mask = (cache.sub_ids == sid)
        act_els = elements[mask]
        unq = np.unique(act_els)
        concs = {e: np.sum(act_els == e) / len(act_els) for e in unq}

        shell_alphas = []

        for shell_d in data['shells']:
            n_map = data['map'][shell_d]
            alphas = []

            for e1 in unq:
                for e2 in unq:
                    obs, tot = 0, 0
                    global_indices_e1 = np.where((cache.sub_ids == sid) & (elements == e1))[0]
                    for idx in global_indices_e1:
                        neighs = n_map.get(idx, [])
                        if neighs:
                            tot += len(neighs)
                            obs += np.sum(elements[neighs] == e2)

                    if tot > 0 and concs[e2] > 0:
                        alpha = 1.0 - (obs / tot) / concs[e2]
                        alphas.append(alpha)
                    else:
                        alphas.append(0.0)

            if alphas:
                shell_rmse = np.sqrt(np.mean(np.array(alphas) ** 2))
                shell_alphas.append(shell_rmse)
            else:
                shell_alphas.append(0.0)

        sub_rmse = np.mean(shell_alphas) if shell_alphas else 0.0
        sub_score = max(0.0, 1.0 - sub_rmse)

        total_rmse += sub_rmse
        sub_count += 1

        sp_clean = ','.join(sorted([s.split('=')[0] for s in cache.specs[sid].split(',')]))
        label = f"S{sub_count}({sp_clean})"
        breakdown_list.append(f"{label}:{sub_score:.3f}")
        sublattice_scores[label] = sub_score

    avg_rmse = total_rmse / sub_count if sub_count > 0 else 0.0
    final_score = max(0.0, 1.0 - avg_rmse)

    return final_score, avg_rmse, " | ".join(breakdown_list), sublattice_scores


def parse_rndstr_content(content):
    lines = content.strip().split('\n')
    sites = []
    for line in lines[4:]:
        line = line.strip()
        if not line: continue
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            element_spec = parts[3]
            sites.append((x, y, z, element_spec))
    return sites


def create_vasp_content(lattice, atoms_frac, elements, nx, ny, nz, comment="Random Structure"):
    sc_lat = np.array([lattice.matrix[0] * nx, lattice.matrix[1] * ny, lattice.matrix[2] * nz])
    from collections import defaultdict
    grouped = defaultdict(list)
    for frac_pos, elem in zip(atoms_frac, elements): grouped[elem].append(frac_pos)
    sorted_elements = sorted(grouped.keys())
    lines = [comment, "1.0"]
    for vec in sc_lat: lines.append(f" {vec[0]:15.9f} {vec[1]:15.9f} {vec[2]:15.9f}")
    lines.append(" ".join(sorted_elements))
    lines.append(" ".join([str(len(grouped[el])) for el in sorted_elements]))
    lines.append("Direct")
    for el in sorted_elements:
        for pos in grouped[el]:
            p = pos % 1.0
            lines.append(f" {p[0]:15.9f} {p[1]:15.9f} {p[2]:15.9f}")
    return "\n".join(lines)


# ==========================================
#  UI RENDERING
# ==========================================

def render_random_analysis_standalone(working_structure, target_concentrations, transformation_matrix,
                                      use_sublattice_mode, chem_symbols, total_atoms):
    st.subheader("🎲 Random Structure Quality Check")

    with st.expander("ℹ️ What is this feature?", expanded=False):

        st.markdown("""
    ### Purpose
    Quantitatively assess whether a randomly occupied supercell could adequately represents a
    random alloy, or whether SQS optimization is likely required.

    The analysis measures short-range order (SRO) using Warren–Cowley parameters, which
    compare local atomic environments to ideal random-alloy statistics.
    """)

        st.markdown("---")

        st.markdown("""
    ### Methodology

    **Sublattice-resolved analysis**  
    Only chemically active sublattices (those allowing multiple species) are analyzed.
    Fixed sublattices are excluded.

    **Coordination shell identification**  
    For each sublattice:
    1. All pairwise distances between atoms on that sublattice are computed using minimum image convention (MIC) 
    2. Distances are rounded to 3 decimal places to group equivalent neighbors
    3. The three shortest unique distances define the 1st, 2nd, and 3rd coordination shells
    4. A neighbor map is constructed: for each atom and each shell, all neighbors at that distance are stored

    **Neighbor enumeration**  
    For each random configuration, the code explicitly enumerates all neighbor pairs by:
    1. Querying the pre-calculated neighbor map for each atom
    2. Counting the actual neighbors found within each shell
    3. Recording both total neighbor counts and species-specific pair counts
    """)

        st.markdown("---")

        st.markdown("### Pair correlations vs higher-order correlations")

        st.markdown("""
    **This analysis uses pair (2-body) correlations only**

    Warren–Cowley parameters quantify **pair correlations**: the probability of finding 
    species β at a specific distance from species α. Higher-order correlations (triplets, 
    quadruplets, etc.) describe the joint probability of finding specific atomic configurations 
    involving 3+ atoms. In most systems, interatomic interactions are dominated by nearest-neighbor (2-body) 
       effects. If the pair distribution is random, the structure effectively samples the 
       canonical ensemble.
       The goal of this analysis is to determine whether random generation could be adequate, or 
       whether SQS optimization is likely needed. If random structures show:
       - **Low pair SRO** → random generation works, higher-order terms are also random
       - **High pair SRO** → SQS is needed, and SQS optimization will target pair + higher-order correlations

    Triplet and higher-order correlations become important during **SQS optimization**, where 
    the goal is to perfectly reproduce random-alloy statistics for cluster-expansion training. 
    However, for the **initial assessment** of whether random structures are 
    adequate, pair correlations provide sufficient diagnostic information.
    """)

        st.markdown("---")

        st.markdown("### Warren–Cowley short-range order")

        st.markdown(
            "For each coordination shell n and species pair (α, β), the Warren–Cowley parameter is:"
        )

        st.latex(r"\alpha_{\alpha\beta}^{(n)} = 1 - \frac{P_{\alpha\beta}^{(n)}}{c_\beta}")

        st.markdown("""
    Where:
    - α is the central atom species
    - β is the neighboring atom species  
    - n is the coordination shell (1st, 2nd, or 3rd nearest neighbor)
    """)

        st.markdown("---")

        st.markdown("### Probability calculation")

        st.latex(r"P_{\alpha\beta}^{(n)} = \frac{N_{\alpha\beta}^{(n)}}{N_{\alpha}^{(n)}}")

        st.markdown("""
    Where:
    """)

        st.latex(
            r"N_{\alpha\beta}^{(n)} = \sum_{i \in \alpha} |\{j \in \text{shell}_n(i) : \text{species}(j) = \beta\}|")

        st.markdown("""
    is the total count of α–β pairs: for each atom i of species α, count how many of its
    neighbors in shell n are species β, then sum over all α atoms.
    """)

        st.latex(r"N_{\alpha}^{(n)} = \sum_{i \in \alpha} |\text{shell}_n(i)|")

        st.markdown("""
    is the total number of neighbors in shell n around all α atoms: for each atom i of species α,
    count how many neighbors exist in shell n, then sum over all α atoms.
    """)

        st.markdown("""
    The code computes these by iterating over all atoms of species α and summing:
    - `obs` (observed β neighbors) = N_αβ^(n)  
    - `tot` (total neighbors) = N_α^(n)
    """)

        st.latex(r"c_\beta = \frac{\text{number of β atoms on sublattice}}{\text{total atoms on sublattice}}")

        st.markdown("""
    is the global concentration of species β on the analyzed sublattice.
    """)

        st.markdown("---")

        st.markdown("### Random alloy reference")

        st.markdown("""
    For a perfectly random alloy, each neighbor position has an independent probability c_β 
    of being occupied by species β. Therefore:
    """)

        st.latex(r"P_{\alpha\beta}^{(n)} = c_\beta \quad \Rightarrow \quad \alpha_{\alpha\beta}^{(n)} = 0")

        st.markdown("---")

        st.markdown("### Interpretation")

        st.markdown("""
    - **α_αβ^(n) > 0**: Species α and β avoid each other in shell n (ordering/segregation)
    - **α_αβ^(n) < 0**: Species α and β prefer pairing in shell n (clustering)  
    - **α_αβ^(n) ≈ 0**: Random-alloy behavior in shell n
    """)

        st.markdown("---")

        st.markdown("### Aggregate metrics")

        st.markdown("""
    For each sublattice and shell, the root-mean-square Warren–Cowley parameter is:
    """)

        st.latex(
            r"\text{RMSE}_{\text{shell}} = \sqrt{\frac{1}{|\text{pairs}|} \sum_{\alpha,\beta} \left(\alpha_{\alpha\beta}^{(n)}\right)^2}")

        st.markdown("""
    The sublattice RMSE averages over its three coordination shells:
    """)

        st.latex(r"\text{RMSE}_{\text{sublattice}} = \frac{1}{3} \sum_{n=1}^{3} \text{RMSE}_{\text{shell},n}")

        st.markdown("""
    The overall structure RMSE averages over all active sublattices:
    """)

        st.latex(
            r"\text{RMSE}_{\text{overall}} = \frac{1}{N_{\text{sublattices}}} \sum_{\text{sublattices}} \text{RMSE}_{\text{sublattice}}")

        st.markdown("""
    The score is:
    """)

        st.latex(r"\text{Score} = 1.0 - \text{RMSE}_{\text{overall}}")

        st.markdown("---")

        st.markdown("### Interpretation guidelines")

        st.markdown("""
    - **Score > 0.95**: Great random-alloy behavior (negligible short-range order), SQS could not be needed.
    - **0.85 < Score ≤ 0.95**: Good randomness; random structures likely adequate, but SQS could improve precision and is recommended.
    - **Score ≤ 0.85**: Ordering, SQS is likely needed.

    **Note**: Smaller supercells naturally exhibit larger statistical fluctuations in Warren-Cowley
    parameters even for truly random configurations. Consider supercell size when interpreting results.
    """)

        st.markdown("---")

    col_config, col_run_it, col_script = st.columns([1, 2, 2])

    with col_config:
        n_random_structures = st.number_input(
            "Number of random structures:", min_value=5, max_value=500, value=20, step=5,
            key="n_random_check_live"
        )
        nx, ny, nz = int(transformation_matrix[0, 0]), int(transformation_matrix[1, 1]), int(
            transformation_matrix[2, 2])
        st.info(f"Supercell: {nx}×{ny}×{nz} | Total Atoms: {total_atoms}")

    with col_script:
        if st.button("📥 Generate Standalone Script", use_container_width=True):
            script = generate_random_analysis_script_standalone(
                working_structure, target_concentrations, transformation_matrix,
                use_sublattice_mode, chem_symbols, n_random_structures
            )
            st.session_state['generated_random_script'] = script

        if 'generated_random_script' in st.session_state:
            st.download_button(
                "💾 Download .sh Script",
                data=st.session_state['generated_random_script'],
                file_name="random_analysis_standalone.sh",
                mime="text/plain",
                use_container_width=True,
                type = 'primary'
            )
            with st.expander("Script Preview"):
                st.code(st.session_state['generated_random_script'], language="bash")

    with col_run_it:
        run_clicked = st.button(
            "🚀 Run Live Analysis",
            type="primary",
            use_container_width=True
        )

    if run_clicked:
        run_live_analysis_logic(
            working_structure, target_concentrations, transformation_matrix,
            use_sublattice_mode, chem_symbols, n_random_structures
        )
    if 'analysis_results' in st.session_state:
        display_results()


def run_live_analysis_logic(working_structure, target_concentrations, transformation_matrix, use_sublattice_mode,
                            chem_symbols, n_random):
    from atat_module import generate_atat_rndstr_content_corrected
    rndstr_content = generate_atat_rndstr_content_corrected(
        working_structure, target_concentrations, use_sublattice_mode, chem_symbols, transformation_matrix
    )
    sites = parse_rndstr_content(rndstr_content)

    status_text = st.empty()
    status_text.info("⚙️ Initializing geometry (calculating shells & neighbors with PBC)...")
    nx, ny, nz = int(transformation_matrix[0, 0]), int(transformation_matrix[1, 1]), int(transformation_matrix[2, 2])

    try:
        cache = GeometryCacheUI(working_structure.lattice, sites, nx, ny, nz)
        status_text.success("✅ Geometry initialized!")
    except Exception as e:
        st.error(f"Geometry init failed: {e}")
        return

    progress_bar = st.progress(0)
    results = []
    structures_data = []
    all_sublattice_keys = set()

    for i in range(n_random):
        seed = random.randint(1, 99999)
        status_text.text(f"Analyzing structure {i + 1}/{n_random}...")

        elements = cache.generate_random_config(seed)
        score, rmse, brk, sub_scores = perform_live_analysis_rigorous(cache, elements)

        vasp_str = create_vasp_content(working_structure.lattice, cache.atoms_frac, elements, nx, ny, nz,
                                       f"Random_{i + 1}_Score_{score:.4f}")

        record = {'ID': i + 1, 'Seed': seed, 'Overall Score': score, 'RMSE': rmse, 'Breakdown': brk}
        for k, v in sub_scores.items():
            record[k] = v
            all_sublattice_keys.add(k)

        structures_data.append({
            'record': record,
            'vasp_content': vasp_str,
            'elements': elements
        })
        results.append(record)
        progress_bar.progress((i + 1) / n_random)

    progress_bar.empty()
    status_text.empty()
    st.session_state['analysis_results'] = {
        'df': pd.DataFrame(results),
        'structures_data': structures_data,
        'all_sublattice_keys': sorted(all_sublattice_keys),
        'cache': cache,
        'debug_info': cache.debug_info if hasattr(cache, 'debug_info') else []
    }


def display_results():
    if 'analysis_results' not in st.session_state:
        return

    res = st.session_state['analysis_results']
    df = res['df']
    structures_data = res['structures_data']
    all_sublattice_keys = res['all_sublattice_keys']
    debug_info = res.get('debug_info', [])

    if debug_info:
        st.markdown("##### 🔍 Geometry Configuration")
        st.code("\n".join(debug_info), language="text")


    avg_score = df['Overall Score'].mean()
    std_score = df['Overall Score'].std()
    best_score = df['Overall Score'].max()
    avg_rmse = df['RMSE'].mean()
    std_rmse = df['RMSE'].std()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Score", f"{avg_score:.4f} ± {std_score:.4f}")
    col2.metric("Mean RMSE", f"{avg_rmse:.4f} ± {std_rmse:.4f}")
    col3.metric("Best Score", f"{best_score:.4f}")

    status_values = f"Mean Score: {avg_score:.4f} ± {std_score:.4f}"

    if avg_score > 0.95:
        st.success(
            f"### ✅ EXCELLENT RANDOMNESS ({status_values})\n**Condition:** Score > 0.95\n\n**Result:** Random generation is sufficient. SQS is likely not needed.")
    elif avg_score > 0.85:
        st.info(
            f"### ☑️ GOOD RANDOMNESS ({status_values})\n**Condition:** 0.85 < Score ≤ 0.95\n\n**Result:** Minor ordering detected. Suggesting to continue with running ATAT mcsqs, **SQS will likely improve precision.**")
    else:
        st.error(
            f"### ⚠️ SIGNIFICANT ORDERING ({status_values})\n**Condition:** Score ≤ 0.85\n\n**Result:** Clustering is likely. **SQS optimization is recommended.** Continue with running ATAT mcsqs.")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Score Distribution (Counts)", "Structure Scores"))

    fig.add_trace(go.Histogram(x=df['Overall Score'], name='Overall Avg', opacity=0.8, marker_color='#2E86C1',
                               visible='legendonly'), row=1, col=1)

    colors = ['#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C', '#34495E']

    for idx, sub_key in enumerate(all_sublattice_keys):
        if sub_key in df.columns:
            fig.add_trace(
                go.Histogram(x=df[sub_key], name=sub_key, opacity=0.6, marker_color=colors[idx % len(colors)]),
                row=1, col=1)

    fig.update_layout(barmode='overlay', bargap=0.2)
    fig.add_vline(x=1.0, line_dash="dash", line_color="green", annotation_text="Excellent", row=1, col=1)
    fig.add_vline(x=0.95, line_dash="dash", line_color="purple", annotation_text="Good", row=1, col=1)
    fig.add_vline(x=0.85, line_dash="dash", line_color="purple", annotation_text="Needs SQS", row=1, col=1)

    fig.add_trace(go.Scatter(x=df['ID'], y=df['Overall Score'], mode='lines+markers', name='Overall Avg',
                             line=dict(color='#2E86C1', width=3), marker=dict(size=10), visible='legendonly'), row=1,
                  col=2)

    for idx, sub_key in enumerate(all_sublattice_keys):
        if sub_key in df.columns:
            fig.add_trace(go.Scatter(x=df['ID'], y=df[sub_key], mode='lines+markers', name=sub_key,
                                     line=dict(color=colors[idx % len(colors)], width=1.0, dash='dot'),
                                     marker=dict(size=10, opacity=0.7)), row=1, col=2)

    fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Excellent",
                  annotation_position="bottom right", row=1, col=2)
    fig.add_hline(y=0.95, line_dash="dash", line_color="purple", annotation_text="Good",
                  annotation_position="bottom right", row=1, col=2)
    fig.add_hline(y=0.85, line_dash="dash", line_color="orange", annotation_text="Needs SQS",
                  annotation_position="bottom right", row=1, col=2)

    fig.update_layout(
        height=600,
        font=dict(family="Arial", size=20, color="black"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=22)),
        plot_bgcolor='rgba(245,245,245,0.5)',
        margin=dict(l=60, r=40, t=60, b=80)
    )
    fig.update_xaxes(title_text="Randomness Score", title_font=dict(size=24), tickfont=dict(size=20), row=1, col=1)
    fig.update_yaxes(title_text="Count", title_font=dict(size=24), tickfont=dict(size=20), row=1, col=1)
    fig.update_xaxes(title_text="Structure ID", title_font=dict(size=24), tickfont=dict(size=20), row=1, col=2)
    fig.update_yaxes(title_text="Score", title_font=dict(size=24), tickfont=dict(size=20), row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 View Detailed Data"):
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📦 Download Best Structures")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_mode = st.radio(
            "**Sort by:**",
            ["Overall Score", "Specific Sublattice"],
            key="filter_mode_display"
        )

    with col2:
        if filter_mode == "Specific Sublattice":
            selected_sublattice = st.selectbox(
                "**Select sublattice:**",
                all_sublattice_keys,
                key="selected_sublattice_display"
            )
        else:
            selected_sublattice = None
            st.write("")

    with col3:
        max_download = st.number_input(
            "**Number of structures:**",
            min_value=1,
            max_value=len(df),
            value=min(10, len(df)),
            step=1,
            key="max_download_display"
        )

    if filter_mode == "Overall Score":
        sort_column = 'Overall Score'
    else:
        sort_column = selected_sublattice if selected_sublattice else 'Overall Score'

    sorted_df = df.sort_values(by=sort_column, ascending=False).head(max_download)

    st.markdown(f"**Preview: Top 3 structures (sorted by {sort_column})**")
    preview_cols = ['ID', 'Overall Score', 'RMSE'] + [k for k in all_sublattice_keys if k in sorted_df.columns]
    st.dataframe(sorted_df[preview_cols].head(3), use_container_width=True)

    if st.button("🔽 Generate Download Package", type="primary", use_container_width=True, key="generate_download_btn"):
        with st.spinner("Creating ZIP package..."):
            zip_buffer = create_filtered_zip(
                sorted_df, structures_data, df,
                filter_mode, sort_column, max_download
            )
            st.session_state['download_zip'] = zip_buffer
            st.session_state['download_count'] = len(sorted_df)
            st.session_state['download_sort_column'] = sort_column
        st.success(f"✅ Package ready with {len(sorted_df)} structures!")

    if 'download_zip' in st.session_state:
        sort_info = st.session_state.get('download_sort_column', 'Overall Score')
        st.download_button(
            f"💾 Download ZIP ({st.session_state['download_count']} structures, sorted by {sort_info})",
            data=st.session_state['download_zip'],
            file_name=f"best_{st.session_state['download_count']}_structures.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_zip_btn",
            type = 'primary'
        )


def create_filtered_zip(sorted_df, structures_data, full_df, filter_mode, sort_column, max_download):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        struct_map = {s['record']['ID']: s for s in structures_data}

        for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
            struct_id = row['ID']
            score = row[sort_column]

            score_str = f"{int(score * 10000):04d}"
            if score >= 1.0:
                score_str = "1_0000"

            filename = f"{score_str}_{rank}.vasp"

            if struct_id in struct_map:
                zf.writestr(f"best_structures/{filename}", struct_map[struct_id]['vasp_content'])

        zf.writestr("analysis_results_all.csv", full_df.to_csv(index=False))
        zf.writestr("analysis_results_selected.csv", sorted_df.to_csv(index=False))

        info_text = f"""Download Package Information
============================

Filter Settings:
- Sort by: {sort_column}
- Number of structures: {max_download}
- Total generated: {len(full_df)}
- Included in package: {len(sorted_df)}

File Naming Convention:
- Format: SCORE_RANK.vasp
- Example: 0_8523_1.vasp means score=0.8523, rank #1
- Score format: 0_XXXX where XXXX = score × 10000

Contents:
- best_structures/ : {len(sorted_df)} POSCAR files
- analysis_results_all.csv : Full analysis results
- analysis_results_selected.csv : Selected structures only
- download_info.txt : This file

Top 5 Structures:
"""
        for rank, (_, row) in enumerate(sorted_df.head(5).iterrows(), 1):
            score_val = row[sort_column]
            info_text += f"{rank}. ID={row['ID']}, Score={score_val:.4f}, RMSE={row['RMSE']:.4f}\n"

        zf.writestr("download_info.txt", info_text)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ==========================================
#  STANDALONE SCRIPT GENERATOR
# ==========================================
def generate_random_analysis_script_standalone(working_structure, target_concentrations, transformation_matrix,
                                               use_sublattice_mode, chem_symbols, n_random_structures):
    nx, ny, nz = int(transformation_matrix[0, 0]), int(transformation_matrix[1, 1]), int(transformation_matrix[2, 2])
    supercell_multiplicity = nx * ny * nz

    from atat_module import generate_atat_rndstr_content_corrected
    rndstr_content = generate_atat_rndstr_content_corrected(
        working_structure, target_concentrations, use_sublattice_mode, chem_symbols, transformation_matrix
    )
    lattice = working_structure.lattice

    script_lines = [
        "#!/bin/bash",
        "# Random Structure Analysis - Standalone",
        f"# Supercell: {nx}×{ny}×{nz} ({supercell_multiplicity} multiplicity)",
        "set -e",
        f'N_RANDOM={n_random_structures}',
        "",
        'echo "========================================"',
        'echo "Random Structure Analysis (Physics-Based)"',
        'echo "========================================"',
        'echo "1. Pre-calculate neighbor lists (PBC-aware)"',
        'echo "2. Detect coordination shells (1st, 2nd, 3rd)"',
        'echo "3. Generate random structures (Exact Stoichiometry) & calculate exact WC parameters"',
        'echo ""',
        "cat > rndstr.in << 'EOF'",
        rndstr_content,
        "EOF",
        "",
        "cat > analyze_random_structure.py << 'PYSCRIPT'",
        "import sys, numpy as np, random",
        "from collections import defaultdict",
        "",
        "def get_mic_distance_matrix(frac_coords, lattice_matrix):",
        "    n = len(frac_coords)",
        "    G = np.dot(lattice_matrix, lattice_matrix.T)",
        "    rows, cols, dists = [], [], []",
        "    for i in range(n):",
        "        d_frac = frac_coords - frac_coords[i]",
        "        d_frac -= np.round(d_frac)",
        "        d2 = np.sum(np.dot(d_frac, G) * d_frac, axis=1)",
        "        d = np.sqrt(d2)",
        "        mask = (d > 1e-4) & (d < 8.0)",
        "        indices = np.where(mask)[0]",
        "        if len(indices) > 0:",
        "            rows.extend([i] * len(indices))",
        "            cols.extend(indices)",
        "            dists.extend(d[indices])",
        "    return np.array(rows), np.array(cols), np.array(dists)",
        "",
        "class GeometryCache:",
        "    def __init__(self, lattice, sites, nx, ny, nz):",
        "        self.specs = sorted(list(set(s[3] for s in sites)))",
        "        self.spec_map = {s: i for i, s in enumerate(self.specs)}",
        "        atoms_frac, sub_ids = [], []",
        "        for ix in range(nx):",
        "            for iy in range(ny):",
        "                for iz in range(nz):",
        "                    for x, y, z, spec in sites:",
        "                        atoms_frac.append([(x+ix)/nx, (y+iy)/ny, (z+iz)/nz])",
        "                        sub_ids.append(self.spec_map[spec])",
        "        self.atoms_frac = np.array(atoms_frac)",
        "        self.sub_ids = np.array(sub_ids)",
        "        self.lattice_matrix = np.array([lattice[0]*nx, lattice[1]*ny, lattice[2]*nz])",
        "        self.n_atoms = len(atoms_frac)",
        "        self.sublattice_shells = {}",
        "        print('DEBUG_INFO: --- Geometry Analysis (PBC) ---')",
        "        for sid, sp in enumerate(self.specs):",
        "            if ',' not in sp: continue",
        "            mask = (self.sub_ids == sid); indices = np.where(mask)[0]",
        "            if len(indices) < 2: continue",
        "            sub_frac = self.atoms_frac[indices]",
        "            r, c, d = get_mic_distance_matrix(sub_frac, self.lattice_matrix)",
        "            if len(d) == 0: continue",
        "            d_rounded = np.round(d, 3)",
        "            unique_d = sorted(list(set(d_rounded)))",
        "            shells = [float(x) for x in unique_d[:3]]",
        "            neighbor_map = {shell_d: defaultdict(list) for shell_d in shells}",
        "            for i_local, j_local, dist_val in zip(r, c, d_rounded):",
        "                if dist_val in shells:",
        "                    neighbor_map[dist_val][indices[i_local]].append(indices[j_local])",
        "            self.sublattice_shells[sid] = {'shells': shells, 'map': neighbor_map}",
        "            print(f'DEBUG_INFO: Sublattice {sid+1} [{sp}] ({len(indices)} sites) Shells: {shells}')",
        "",
        "    def generate_random_config(self, seed):",
        "        random.seed(seed); np.random.seed(seed)",
        "        final_elements = [None] * len(self.sub_ids)",
        "        sub_indices = defaultdict(list)",
        "        for idx, sid in enumerate(self.sub_ids): sub_indices[sid].append(idx)",
        "        for sid, indices in sub_indices.items():",
        "            spec = self.specs[sid]",
        "            if '=' in spec:",
        "                els, probs = [], []",
        "                for s in spec.split(','): e, p = s.split('='); els.append(e); probs.append(float(p))",
        "                n_sites = len(indices)",
        "                counts = [int(round(f * n_sites)) for f in probs]",
        "                diff = n_sites - sum(counts)",
        "                if diff != 0: counts[-1] += diff",
        "                atoms = []",
        "                for e, c in zip(els, counts): atoms.extend([e]*c)",
        "                random.shuffle(atoms)",
        "                for i, atom in zip(indices, atoms): final_elements[i] = atom",
        "            else:",
        "                for i in indices: final_elements[i] = spec",
        "        return final_elements",
        "",
        "def analyze_config(cache, elements):",
        "    total_rmse, sub_count, breakdown = 0.0, 0, []",
        "    elements = np.array(elements)",
        "    for sid, data in cache.sublattice_shells.items():",
        "        mask = (cache.sub_ids == sid)",
        "        act_els = elements[mask]",
        "        unq = np.unique(act_els)",
        "        concs = {e: np.sum(act_els==e)/len(act_els) for e in unq}",
        "        shell_alphas = []",
        "        for shell_d in data['shells']:",
        "            n_map = data['map'][shell_d]",
        "            alphas = []",
        "            for e1 in unq:",
        "                for e2 in unq:",
        "                    obs, tot = 0, 0",
        "                    for idx in np.where((cache.sub_ids == sid) & (elements == e1))[0]:",
        "                        neighs = n_map.get(idx, [])",
        "                        if neighs: tot += len(neighs); obs += np.sum(elements[neighs] == e2)",
        "                    if tot > 0 and concs[e2] > 0:",
        "                        alphas.append(1.0 - (obs/tot)/concs[e2])",
        "                    else:",
        "                        alphas.append(0.0)",
        "            if alphas:",
        "                shell_rmse = np.sqrt(np.mean(np.array(alphas)**2))",
        "                shell_alphas.append(shell_rmse)",
        "            else:",
        "                shell_alphas.append(0.0)",
        "        sub_rmse = np.mean(shell_alphas) if shell_alphas else 0.0",
        "        sub_score = max(0.0, 1.0 - sub_rmse)",
        "        total_rmse += sub_rmse; sub_count += 1",
        "        sp_clean = ','.join(sorted([s.split('=')[0] for s in cache.specs[sid].split(',')]))",
        "        breakdown.append(f'S{sub_count}({sp_clean}):{sub_score:.3f}')",
        "    avg_rmse = total_rmse / sub_count if sub_count > 0 else 0.0",
        "    final_score = max(0.0, 1.0 - avg_rmse)",
        "    return final_score, avg_rmse, ' | '.join(breakdown)",
        "",
        "def parse_rndstr():",
        "    with open('rndstr.in', 'r') as f: lines = f.readlines()",
        f"    a, b, c = {lattice.a:.6f}, {lattice.b:.6f}, {lattice.c:.6f}",
        f"    alpha, beta, gamma = {lattice.alpha:.2f}, {lattice.beta:.2f}, {lattice.gamma:.2f}",
        "    from math import cos, sin, radians, sqrt",
        "    alpha_r, beta_r, gamma_r = map(radians, [alpha, beta, gamma])",
        "    lat = np.array([",
        "        [a, 0, 0],",
        "        [b * cos(gamma_r), b * sin(gamma_r), 0],",
        "        [c * cos(beta_r), ",
        "         c * (cos(alpha_r) - cos(beta_r) * cos(gamma_r)) / sin(gamma_r),",
        "         c * sqrt(1 - cos(beta_r)**2 - ((cos(alpha_r) - cos(beta_r) * cos(gamma_r)) / sin(gamma_r))**2)]",
        "    ])",
        "    sites = []",
        "    for l in lines[4:]: ",
        "        if l.strip(): p=l.split(); sites.append((float(p[0]), float(p[1]), float(p[2]), p[3]))",
        "    return lat, sites",
        "",
        "def write_vasp(fname, lattice, atoms_frac, elements, comm):",
        "    from collections import defaultdict",
        "    g = defaultdict(list)",
        "    for p, e in zip(atoms_frac, elements): g[e].append(p)",
        "    els = sorted(g.keys())",
        "    with open(fname, 'w') as f:",
        "        f.write(f'{comm}\\n1.0\\n')",
        "        for v in lattice: f.write(f' {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\\n')",
        "        f.write(' '.join(els)+'\\n' + ' '.join([str(len(g[e])) for e in els]) + '\\nDirect\\n')",
        "        for e in els: ",
        "            for p in g[e]: f.write(f' {p[0]%1:.6f} {p[1]%1:.6f} {p[2]%1:.6f}\\n')",
        "",
        "if __name__ == '__main__':",
        "    mode = sys.argv[1]",
        "    if mode == 'init':",
        "        lat, sites = parse_rndstr()",
        f"        cache = GeometryCache(lat, sites, {nx}, {ny}, {nz})",
        "        import pickle; pickle.dump(cache, open('geo_cache.pkl', 'wb'))",
        "    else:",
        "        import pickle; cache = pickle.load(open('geo_cache.pkl', 'rb'))",
        "        seed = int(mode)",
        "        els = cache.generate_random_config(seed)",
        "        score, rmse, brk = analyze_config(cache, els)",
        "        write_vasp('POSCAR_current.vasp', cache.lattice_matrix, cache.atoms_frac, els, f'Random {seed}')",
        "        print(f'OVERALL_SCORE:{score:.6f}')",
        "        print(f'WC_RMSE:{rmse:.6f}')",
        "        print(f'N_ATOMS:{cache.n_atoms}')",
        "        print(f'BREAKDOWN:{brk}')",
        "PYSCRIPT",
        "",
        'mkdir -p random_structures',
        'echo "RandomID,Seed,Overall_Score,WC_RMSE,N_Atoms,Sublattice_Breakdown" > random_analysis_results.csv',
        "python3 analyze_random_structure.py init",
        "for i in $(seq 1 $N_RANDOM); do",
        "    SEED=$RANDOM",
        '    printf "[%2d/%2d] Structure #%d ... " $i $N_RANDOM $i',
        "    RES=$(python3 analyze_random_structure.py $SEED 2>&1)",
        "    if [ $i -eq 1 ]; then",
        '        echo ""; echo "--- 🔍 CONFIGURATION CHECK (Structure #1) ---"',
        "        echo \"$RES\" | grep 'DEBUG_INFO' | sed 's/DEBUG_INFO: //'",
        '        echo "---------------------------------------------"; printf "[%2d/%2d] Structure #%d ... " $i $N_RANDOM $i',
        "    fi",
        "    SCORE=$(echo \"$RES\" | grep 'OVERALL_SCORE:' | cut -d':' -f2)",
        "    RMSE=$(echo \"$RES\" | grep 'WC_RMSE:' | cut -d':' -f2)",
        "    ATOMS=$(echo \"$RES\" | grep 'N_ATOMS:' | cut -d':' -f2)",
        "    BRK=$(echo \"$RES\" | grep 'BREAKDOWN:' | cut -d':' -f2-)",
        "    if [ -n \"$SCORE\" ]; then",
        "        mv POSCAR_current.vasp \"random_structures/POSCAR_${i}.vasp\"",
        "        echo \"$i,$SEED,$SCORE,$RMSE,$ATOMS,$BRK\" >> random_analysis_results.csv",
        '        echo "✅ $SCORE [$BRK]"',
        "    else echo \"❌ Failed\"; fi",
        "done",
        "",
        "python3 << 'PYSCRIPT'",
        "import csv",
        "try:",
        "    scores = []",
        "    with open('random_analysis_results.csv', 'r') as f:",
        "        for row in csv.DictReader(f): scores.append(float(row['Overall_Score']))",
        "    if scores:",
        "        n = len(scores)",
        "        avg = sum(scores)/n",
        "        std = (sum([(s-avg)**2 for s in scores])/n)**0.5",
        "        print(f'\\n=== Summary ===\\nMean Score: {avg:.4f} ± {std:.4f}')",
        "        if avg > 0.95: print('✅ EXCELLENT - Random is sufficient')",
        "        elif avg > 0.85: print('☑️  GOOD - SQS will likely improve precision')",
        "        else: print('⚠️  SIGNIFICANT ORDERING - Use SQS')",
        "except: pass",
        "PYSCRIPT"
    ]
    return '\n'.join(script_lines)
