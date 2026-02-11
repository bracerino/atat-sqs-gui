import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def identify_sublattices(structure, chem_symbols):
    sga = SpacegroupAnalyzer(structure)
    wyckoff_symbols = sga.get_symmetry_dataset().wyckoffs

    mixed_occupancy_sites = []
    for i, site_elements in enumerate(chem_symbols):
        if len(site_elements) >= 2:
            mixed_occupancy_sites.append(i)

    if not mixed_occupancy_sites:
        return {}

    wyckoff_to_sites = {}
    for i, ws in enumerate(wyckoff_symbols):
        if ws not in wyckoff_to_sites:
            wyckoff_to_sites[ws] = []
        wyckoff_to_sites[ws].append(i)

    sublattices = {}
    processed = set()
    for site_idx in mixed_occupancy_sites:
        ws = wyckoff_symbols[site_idx]
        if ws not in processed:
            sites = [s for s in wyckoff_to_sites[ws] if s in mixed_occupancy_sites]
            elements = "/".join(sorted(set(str(e) for s in sites for e in chem_symbols[s])))
            label = f"Wyckoff {ws} ({elements}, {len(sites)} sites)"
            sublattices[label] = sites
            processed.add(ws)

    return sublattices


def calculate_pair_distances_histogram(structure, chem_symbols=None, use_sublattice_mode=False,
                                       distance_precision=4, max_distance=12.0,
                                       forced_active_sites=None):
    from pymatgen.core.lattice import Lattice

    original_lattice = structure.lattice
    a, b, c = original_lattice.abc
    max_param = max(a, b, c)

    normalized_lattice = Lattice.from_parameters(
        a / max_param, b / max_param, c / max_param,
        original_lattice.alpha, original_lattice.beta, original_lattice.gamma
    )

    normalized_structure = structure.copy()
    normalized_structure.lattice = normalized_lattice

    active_sites = []

    if forced_active_sites is not None:
        active_sites = list(forced_active_sites)
    elif use_sublattice_mode and chem_symbols:
        sga = SpacegroupAnalyzer(normalized_structure)
        wyckoff_symbols = sga.get_symmetry_dataset().wyckoffs

        mixed_occupancy_sites = []
        for i, site_elements in enumerate(chem_symbols):
            if len(site_elements) >= 2:
                mixed_occupancy_sites.append(i)

        if not mixed_occupancy_sites:
            return {
                'distances': [],
                'normalized_distances': [],
                'multiplicities': [],
                'message': "No mixed-occupancy sites found for histogram generation.",
                'active_sites': [],
                'max_param': max_param
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
                mixed_sites_with_same_wyckoff = [s for s in sites_with_same_wyckoff if s in mixed_occupancy_sites]
                for equiv_site in mixed_sites_with_same_wyckoff:
                    if equiv_site not in active_sites:
                        active_sites.append(equiv_site)
                wyckoff_positions_processed.add(wyckoff_symbol)
    else:
        active_sites = list(range(len(normalized_structure.sites)))

    if not active_sites:
        return {
            'distances': [],
            'normalized_distances': [],
            'multiplicities': [],
            'message': "No active sites found.",
            'active_sites': [],
            'max_param': max_param
        }

    active_lattice = normalized_structure.lattice
    active_species = []
    active_coords = []

    for site_idx in active_sites:
        site = normalized_structure[site_idx]
        active_species.append(site.specie)
        active_coords.append(site.frac_coords)

    active_structure = Structure(
        lattice=active_lattice,
        species=active_species,
        coords=active_coords,
        coords_are_cartesian=False
    )

    active_supercell = active_structure * (5, 5, 5)

    original_active_sites = len(active_structure)
    center_cell_index = 62
    center_cell_start = center_cell_index * original_active_sites
    center_cell_end = center_cell_start + original_active_sites

    all_distances = []

    for i in range(center_cell_start, center_cell_end):
        center_site = active_supercell[i]

        for j in range(len(active_supercell)):
            if i == j:
                continue

            if center_cell_start <= j < center_cell_end:
                if j <= i:
                    continue

            distance = center_site.distance(active_supercell[j])

            if 0.001 < distance < max_distance / max_param:
                all_distances.append(distance)

    if not all_distances:
        return {
            'distances': [],
            'normalized_distances': [],
            'multiplicities': [],
            'message': "No pair distances calculated.",
            'active_sites': active_sites,
            'max_param': max_param
        }

    all_distances = [round(d, distance_precision) for d in all_distances]

    distance_counts = {}
    for d in all_distances:
        distance_counts[d] = distance_counts.get(d, 0) + 1

    sorted_distances = sorted(distance_counts.keys())
    sorted_multiplicities = [distance_counts[d] for d in sorted_distances]

    total_pairs = len(all_distances)
    unique_distances = len(sorted_distances)

    distances_angstrom = [d * max_param for d in sorted_distances]

    return {
        'distances': distances_angstrom,
        'normalized_distances': sorted_distances,
        'multiplicities': sorted_multiplicities,
        'distance_counts_angstrom': {d * max_param: distance_counts[d] for d in sorted_distances},
        'distance_counts_normalized': distance_counts,
        'active_sites': active_sites,
        'total_sites': len(normalized_structure.sites),
        'total_pairs': total_pairs,
        'unique_distances': unique_distances,
        'distance_precision': distance_precision,
        'max_param': max_param,
        'lattice_abc': (a, b, c),
        'message': f"Successfully calculated {total_pairs} unique pair distances ({unique_distances} unique values) from {len(active_sites)} active sites"
    }


def create_pair_distance_histogram_plot(histogram_data, use_normalized=False):
    if not histogram_data['distances']:
        return None

    if use_normalized:
        distances = histogram_data['normalized_distances']
        x_label = "Pair Distance (normalized to max lattice parameter)"
        max_param = histogram_data['max_param']
        hover_template = (f'<b style="font-size:18px">Distance: %{{x:.4f}} (= %{{customdata:.4f}} Å)</b><br>'
                          '<b style="font-size:18px">Multiplicity: %{y}</b><br>'
                          '<extra></extra>')
        customdata = histogram_data['distances']
        x_range = [0, 1.5]
    else:
        distances = histogram_data['distances']
        x_label = "Pair Distance (Å)"
        max_param = histogram_data['max_param']
        hover_template = ('<b style="font-size:18px">Distance: %{x:.4f} Å</b><br>'
                          f'<b style="font-size:18px">Normalized: %{{customdata:.4f}}</b><br>'
                          '<b style="font-size:18px">Multiplicity: %{y}</b><br>'
                          '<extra></extra>')
        customdata = histogram_data['normalized_distances']
        x_range = None

    multiplicities = histogram_data['multiplicities']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=distances,
        y=multiplicities,
        name='Pair Multiplicity',
        customdata=customdata,
        marker=dict(
            color=multiplicities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Multiplicity",
                    font=dict(size=20, family="Arial Black")
                ),
                tickfont=dict(size=18),
                thickness=25,
                len=0.7
            )
        ),
        hovertemplate=hover_template,
        width=0.01
    ))

    fig.update_layout(
        title=dict(
            text="Pair-Distance Histogram (Number of Pairs vs Distance)",
            font=dict(size=28, family="Arial Black", color='#1e3a8a')
        ),
        xaxis_title=x_label,
        yaxis_title="Number of Pairs (Multiplicity)",
        hovermode='closest',
        height=700,
        showlegend=False,
        font=dict(size=20, family="Arial"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=2,
            title_font=dict(size=24, family="Arial", color='#1e3a8a'),
            tickfont=dict(size=20, color='black'),
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor='black',
            showline=True,
            linewidth=3,
            linecolor='black',
            mirror=True,
            range=x_range
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            gridwidth=2,
            title_font=dict(size=24, family="Arial", color='#1e3a8a'),
            tickfont=dict(size=20, color='black'),
            zeroline=True,
            zerolinewidth=3,
            zerolinecolor='black',
            showline=True,
            linewidth=3,
            linecolor='black',
            mirror=True
        )
    )

    return fig


def suggest_cutoff_distances(histogram_data, num_suggestions=6, use_normalized=False):
    if not histogram_data['distances'] or len(histogram_data['distances']) < 2:
        return []

    if use_normalized:
        distances = histogram_data['normalized_distances']
    else:
        distances = histogram_data['distances']

    multiplicities = histogram_data['multiplicities']

    gaps = []
    for i in range(len(distances) - 1):
        distance_gap = distances[i + 1] - distances[i]
        current_multiplicity = multiplicities[i]
        next_multiplicity = multiplicities[i + 1]
        avg_multiplicity = (current_multiplicity + next_multiplicity) / 2

        if distance_gap > 0.001:
            gap_score = distance_gap / (avg_multiplicity / 100 + 0.01)

            gaps.append({
                'after_distance': distances[i],
                'suggested_cutoff': round((distances[i] + distances[i + 1]) / 2, 4),
                'gap_score': gap_score,
                'pairs_included': sum(multiplicities[:i + 1])
            })

    gaps.sort(key=lambda x: x['gap_score'], reverse=True)

    suggestions = []
    seen_cutoffs = set()

    for gap in gaps:
        cutoff = gap['suggested_cutoff']
        if cutoff not in seen_cutoffs:
            seen_cutoffs.add(cutoff)
            suggestions.append({
                'cutoff': cutoff,
                'pairs_included': gap['pairs_included'],
                'total_pairs': histogram_data['total_pairs']
            })

        if len(suggestions) >= num_suggestions:
            break

    suggestions.sort(key=lambda x: x['cutoff'])

    return suggestions


def render_pair_distance_histogram_tab(working_structure, chem_symbols, use_sublattice_mode):

    col_prec, col_maxdist, col_btn = st.columns([1, 1, 2])

    with col_prec:
        distance_precision = st.selectbox(
            "Distance precision:",
            options=[3, 4, 5, 6],
            index=1,
            help="Number of decimal places. 4 is recommended.",
            key="histogram_precision"
        )

    with col_maxdist:
        max_distance = st.number_input(
            "Max distance (Å):",
            min_value=5.0,
            max_value=20.0,
            value=12.0,
            step=1.0,
            help="Maximum pair distance to calculate",
            key="histogram_max_dist"
        )

    with col_btn:
        calculate_btn = st.button(
            "📊 Calculate Histogram",
            type="primary",
            key="calc_histogram_atat",
            use_container_width=True
        )

    if calculate_btn:
        if use_sublattice_mode and chem_symbols:
            sublattices = identify_sublattices(working_structure, chem_symbols)

            if not sublattices:
                st.warning("No mixed-occupancy sites found for histogram generation.")
                st.session_state['histogram_data'] = None
                st.session_state['histogram_per_sublattice'] = None
                return

            per_sublattice = {}

            with st.spinner("Calculating combined histogram (all sublattices)..."):
                all_sites = []
                for sites in sublattices.values():
                    all_sites.extend(sites)
                all_sites = sorted(set(all_sites))

                combined_data = calculate_pair_distances_histogram(
                    working_structure,
                    chem_symbols,
                    use_sublattice_mode=False,
                    distance_precision=distance_precision,
                    max_distance=max_distance,
                    forced_active_sites=all_sites
                )
                per_sublattice["All sublattices (combined)"] = combined_data

            for label, sites in sublattices.items():
                with st.spinner(f"Calculating histogram for {label}..."):
                    sub_data = calculate_pair_distances_histogram(
                        working_structure,
                        chem_symbols,
                        use_sublattice_mode=False,
                        distance_precision=distance_precision,
                        max_distance=max_distance,
                        forced_active_sites=sites
                    )
                    per_sublattice[label] = sub_data

            st.session_state['histogram_per_sublattice'] = per_sublattice
            st.session_state['histogram_data'] = None

        else:
            with st.spinner("Calculating pair distances (5×5×5 supercell)..."):
                histogram_data = calculate_pair_distances_histogram(
                    working_structure,
                    chem_symbols if use_sublattice_mode else None,
                    use_sublattice_mode,
                    distance_precision=distance_precision,
                    max_distance=max_distance
                )

            st.session_state['histogram_data'] = histogram_data
            st.session_state['histogram_per_sublattice'] = None

    if use_sublattice_mode and st.session_state.get('histogram_per_sublattice'):
        per_sublattice = st.session_state['histogram_per_sublattice']

        sublattice_labels = list(per_sublattice.keys())
        selected_label = st.selectbox(
            "Select sublattice to display:",
            options=sublattice_labels,
            index=0,
            key="sublattice_selector"
        )

        histogram_data = per_sublattice[selected_label]
        _display_histogram_results(histogram_data, use_sublattice_mode, selected_label)

    elif 'histogram_data' in st.session_state and st.session_state['histogram_data']:
        histogram_data = st.session_state['histogram_data']
        _display_histogram_results(histogram_data, use_sublattice_mode)


def _display_histogram_results(histogram_data, use_sublattice_mode, sublattice_label=None):

    if 'message' in histogram_data and not histogram_data['distances']:
        st.warning(histogram_data['message'])
        return

    st.success(histogram_data['message'])

    if 'active_sites' in histogram_data:
        active_count = len(histogram_data['active_sites'])
        total_count = histogram_data['total_sites']
        a, b, c = histogram_data['lattice_abc']
        max_param = histogram_data['max_param']

        info_text = f"**Lattice:** a={a:.4f} Å, b={b:.4f} Å, c={c:.4f} Å (max={max_param:.4f} Å)"
        if use_sublattice_mode and sublattice_label:
            info_text += f" | **Active sites:** {active_count}/{total_count} ({sublattice_label})"
        elif use_sublattice_mode:
            info_text += f" | **Active sites:** {active_count}/{total_count} (mixed-occupancy only)"
        else:
            info_text += f" | **All {active_count} sites** considered"

        st.info(info_text)

    st.subheader("📊 Pair-Distance Distribution")

    use_normalized = st.toggle(
        "Show normalized distances (divide by max lattice parameter)",
        value=True,
        help="Toggle between actual distances (Å) and normalized distances",
        key="use_normalized_distances"
    )

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("Unique Distances", histogram_data['unique_distances'])
    with col_stat2:
        st.metric("Total Pairs", histogram_data['total_pairs'])
    with col_stat3:
        if use_normalized:
            st.metric("Min Distance", f"{min(histogram_data['normalized_distances']):.4f}")
        else:
            st.metric("Min Distance", f"{min(histogram_data['distances']):.4f} Å")
    with col_stat4:
        if use_normalized:
            st.metric("Max Distance", f"{max(histogram_data['normalized_distances']):.4f}")
        else:
            st.metric("Max Distance", f"{max(histogram_data['distances']):.4f} Å")

    fig = create_pair_distance_histogram_plot(histogram_data, use_normalized=use_normalized)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 First 15 Distance Shells", expanded=False):
        detail_data = []
        n_show = min(15, len(histogram_data['distances']))

        for i in range(n_show):
            if use_normalized:
                dist_str = f"{histogram_data['normalized_distances'][i]:.4f}"
                angstrom_equiv = f"({histogram_data['distances'][i]:.4f} Å)"
            else:
                dist_str = f"{histogram_data['distances'][i]:.4f} Å"
                angstrom_equiv = f"({histogram_data['normalized_distances'][i]:.4f} norm.)"

            detail_data.append({
                "Shell": i + 1,
                "Distance": dist_str,
                "Equivalent": angstrom_equiv,
                "Multiplicity": histogram_data['multiplicities'][i],
                "Cumulative Pairs": sum(histogram_data['multiplicities'][:i + 1])
            })

        detail_df = pd.DataFrame(detail_data)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.subheader("💡 Suggested Cluster Cutoff Distances")
    suggestions = suggest_cutoff_distances(histogram_data, num_suggestions=8, use_normalized=use_normalized)

    if suggestions:
        suggestion_data = []
        for suggestion in suggestions:
            if use_normalized:
                cutoff_str = f"{suggestion['cutoff']:.4f}"
            else:
                cutoff_str = f"{suggestion['cutoff']:.4f} Å"

            suggestion_data.append({
                "Suggested Cutoff": cutoff_str,
                "Cumulative Pairs": suggestion['pairs_included']
            })

        suggestion_df = pd.DataFrame(suggestion_data)
        st.dataframe(suggestion_df, use_container_width=True, hide_index=True)

    else:
        st.warning("No significant gaps found in the distance distribution.")

    st.subheader("📥 Download Data")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if use_normalized:
            csv_data = pd.DataFrame({
                'Distance (normalized)': histogram_data['normalized_distances'],
                'Distance (Angstrom)': histogram_data['distances'],
                'Multiplicity': histogram_data['multiplicities']
            })
        else:
            csv_data = pd.DataFrame({
                'Distance (Angstrom)': histogram_data['distances'],
                'Distance (normalized)': histogram_data['normalized_distances'],
                'Multiplicity': histogram_data['multiplicities']
            })

        csv_string = csv_data.to_csv(index=False)

        dl_key_suffix = ""
        if sublattice_label:
            dl_key_suffix = "_" + sublattice_label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_")

        st.download_button(
            label="📥 Download Histogram CSV",
            data=csv_string,
            file_name="pair_distance_histogram.csv",
            mime="text/csv",
            key=f"download_histogram_csv{dl_key_suffix}",
            use_container_width=True,
            type = "primary"
        )

    with col_dl2:
        if suggestions:
            suggestions_csv = pd.DataFrame(suggestion_data).to_csv(index=False)
            st.download_button(
                label="📥 Download Suggestions CSV",
                data=suggestions_csv,
                file_name="cutoff_suggestions.csv",
                mime="text/csv",
                key=f"download_suggestions_csv{dl_key_suffix}",
                use_container_width=True,
                type="primary"
            )
