import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go


def render_parallel_csv_analysis_tab():
    st.write("**Upload mcsqs_parallel_progress.csv file to analyze parallel runs:**")
    uploaded_parallel_csv = st.file_uploader(
        "Upload mcsqs_parallel_progress.csv file:",
        type=['csv'],
        help="Upload the mcsqs_parallel_progress.csv file generated by the parallel monitoring script",
        key="parallel_csv_uploader"
    )

    if uploaded_parallel_csv is not None:
        try:
            csv_content = uploaded_parallel_csv.read().decode('utf-8')
            is_valid, validation_message, parallel_data = validate_parallel_csv(csv_content)

            if not is_valid:
                st.error(f"Invalid parallel CSV file: {validation_message}")
                return


            minutes, run_data, best_objectives, best_runs = parallel_data
            num_runs = len(run_data)

            st.success(f"✅ Successfully parsed {len(minutes)} time points from {num_runs} parallel runs!")

            final_objectives = {}
            best_overall_objectives = {}
            total_steps = {}

            for run_id in run_data:
                objectives = [obj for obj in run_data[run_id]['objectives'] if obj != 'N/A' and obj is not None]
                steps = [step for step in run_data[run_id]['steps'] if step != 'N/A' and step is not None]

                if objectives:
                    final_objectives[run_id] = objectives[-1]
                    best_overall_objectives[run_id] = min(objectives)
                if steps:
                    total_steps[run_id] = max(steps)

            if not final_objectives:
                st.warning("No valid objective function data found in the CSV file.")
                return

            best_run_id = min(final_objectives.keys(), key=lambda x: final_objectives[x])
            worst_run_id = max(final_objectives.keys(), key=lambda x: final_objectives[x])

            best_final = final_objectives[best_run_id]
            worst_final = final_objectives[worst_run_id]
            avg_final = sum(final_objectives.values()) / len(final_objectives)

            best_overall = min(best_overall_objectives.values())
            avg_overall = sum(best_overall_objectives.values()) / len(best_overall_objectives)

            st.subheader("📊 Parallel Runs Summary")

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Total Runs", num_runs)
                st.metric("Best Final Objective", f"{best_final:.6f}")

            with col_stat2:
                st.metric("Worst Final Objective", f"{worst_final:.6f}")
                st.metric("Average Final Objective", f"{avg_final:.6f}")

            with col_stat3:
                st.metric("Best Overall Objective", f"{best_overall:.6f}")
                st.metric("Standard Deviation", f"{np.std(list(final_objectives.values())):.6f}")

            with col_stat4:
                st.metric("Objective Range", f"{worst_final - best_final:.6f}")
                st.metric("Runtime", f"{minutes[-1] - minutes[0]:.0f} min")

            st.subheader("🏆 Best vs Worst Runs")

            col_best, col_worst = st.columns(2)

            with col_best:
                st.success(f"**🥇 Best Performing Run: {best_run_id}**")
                st.write(f"**Final Objective:** {final_objectives[best_run_id]:.6f}")
                st.write(f"**Best Objective:** {best_overall_objectives[best_run_id]:.6f}")
                if best_run_id in total_steps:
                    st.write(f"**Total Steps:** {total_steps[best_run_id]}")

                best_objectives_list = [obj for obj in run_data[best_run_id]['objectives'] if
                                        obj != 'N/A' and obj is not None]
                if len(best_objectives_list) > 1:
                    improvement = best_objectives_list[-1] - best_objectives_list[0]
                    st.write(f"**Total Improvement:** {improvement:.6f}")

            with col_worst:
                st.error(f"**🥉 Worst Performing Run: {worst_run_id}**")
                st.write(f"**Final Objective:** {final_objectives[worst_run_id]:.6f}")
                st.write(f"**Best Objective:** {best_overall_objectives[worst_run_id]:.6f}")
                if worst_run_id in total_steps:
                    st.write(f"**Total Steps:** {total_steps[worst_run_id]}")

                worst_objectives_list = [obj for obj in run_data[worst_run_id]['objectives'] if
                                         obj != 'N/A' and obj is not None]
                if len(worst_objectives_list) > 1:
                    improvement = worst_objectives_list[-1] - worst_objectives_list[0]
                    st.write(f"**Total Improvement:** {improvement:.6f}")

            st.subheader("📋 Detailed Comparison")

            comparison_data = []
            for run_id in sorted(run_data.keys()):
                if run_id in final_objectives:
                    performance = "🥇 Best" if run_id == best_run_id else "🥉 Worst" if run_id == worst_run_id else "✅ Good"

                    comparison_data.append({
                        "Run": run_id,
                        "Final Objective": f"{final_objectives[run_id]:.6f}",
                        "Best Objective": f"{best_overall_objectives[run_id]:.6f}",
                        "Total Steps": total_steps.get(run_id, "N/A"),
                        "Performance": performance
                    })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("📈 Optimization Progress Over Time")

            fig = create_parallel_progress_plot(minutes, run_data, best_objectives, best_runs, best_run_id,
                                                worst_run_id)
            st.plotly_chart(fig, use_container_width=True)

        except UnicodeDecodeError:
            st.error("Error reading CSV file. Please ensure the file is a text file with UTF-8 encoding.")
        except Exception as e:
            st.error(f"Error processing parallel CSV file: {str(e)}")
            import traceback
            st.error(f"Debug info: {traceback.format_exc()}")


def validate_parallel_csv(csv_content):
    try:
        df = pd.read_csv(StringIO(csv_content))

        if 'Minute' not in df.columns or 'Timestamp' not in df.columns:
            return False, "Missing required columns: Minute, Timestamp", None

        run_columns = [col for col in df.columns if col.startswith('Run') and col.endswith('_Objective')]
        if not run_columns:
            return False, "No run objective columns found (expected Run1_Objective, Run2_Objective, etc.)", None

        num_runs = len(run_columns)

        if len(df) == 0:
            return False, "CSV file is empty", None

        minutes = df['Minute'].tolist()
        run_data = {}

        for i in range(1, num_runs + 1):
            run_id = f"Run{i}"
            obj_col = f"Run{i}_Objective"
            steps_col = f"Run{i}_Steps"
            status_col = f"Run{i}_Status"

            if obj_col in df.columns:
                objectives = df[obj_col].tolist()
                steps = df[steps_col].tolist() if steps_col in df.columns else [0] * len(objectives)
                statuses = df[status_col].tolist() if status_col in df.columns else ['UNKNOWN'] * len(objectives)

                run_data[run_id] = {
                    'objectives': objectives,
                    'steps': steps,
                    'statuses': statuses
                }

        best_objectives = df['Best_Overall_Objective'].tolist() if 'Best_Overall_Objective' in df.columns else []
        best_runs = df['Best_Run'].tolist() if 'Best_Run' in df.columns else []

        data_points = len(df)
        time_span = df['Minute'].max() - df['Minute'].min()

        return True, f"Valid parallel CSV with {data_points} data points from {num_runs} runs over {time_span:.0f} minutes", (
            minutes, run_data, best_objectives, best_runs)

    except Exception as e:
        return False, f"Error parsing CSV file: {str(e)}", None


def create_parallel_progress_plot(minutes, run_data, best_objectives, best_runs, best_run_id, worst_run_id):
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, (run_id, data) in enumerate(run_data.items()):
        objectives = [obj if obj != 'N/A' and obj is not None else None for obj in data['objectives']]

        color = colors[i % len(colors)]
        line_width = 3 if run_id == best_run_id else 2 if run_id == worst_run_id else 1

        name_suffix = " (Best)" if run_id == best_run_id else " (Worst)" if run_id == worst_run_id else ""

        fig.add_trace(go.Scatter(
            x=minutes,
            y=objectives,
            mode='lines',
            name=f"{run_id}{name_suffix}",
            line=dict(color=color, width=line_width),
            connectgaps=False,
            hovertemplate=f'<b>{run_id}</b><br>Minute: %{{x}}<br>Objective: %{{y:.6f}}<extra></extra>'
        ))


    fig.update_layout(
        title=dict(
            text="Parallel MCSQS Optimization Progress",
            font=dict(size=24, family="Arial Black")
        ),
        xaxis_title="Time (Minutes)",
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

    return fig


def analyze_parallel_convergence(minutes, run_data, best_objectives):
    all_final_objectives = []
    all_improvements = []
    converged_runs = 0

    for run_id, data in run_data.items():
        objectives = [obj for obj in data['objectives'] if obj != 'N/A' and obj is not None]
        if len(objectives) > 5:
            final_quarter = objectives[-len(objectives) // 4:] if len(objectives) >= 4 else objectives[-2:]
            final_std = np.std(final_quarter)

            if final_std < 0.001:
                converged_runs += 1

            all_final_objectives.append(objectives[-1])
            if len(objectives) > 1:
                all_improvements.append(objectives[-1] - objectives[0])

    if not all_final_objectives:
        return {"Convergence Status": "Unknown", "Recommendation": "Insufficient data"}

    avg_final = np.mean(all_final_objectives)
    std_final = np.std(all_final_objectives)
    avg_improvement = np.mean(all_improvements) if all_improvements else 0

    convergence_rate = converged_runs / len(run_data) if run_data else 0

    runtime = minutes[-1] - minutes[0] if len(minutes) > 1 else 0

    if convergence_rate >= 0.8 and std_final < 0.01:
        status = "Excellent"
        recommendation = "Most runs converged well - results are reliable"
    elif convergence_rate >= 0.5 and std_final < 0.05:
        status = "Good"
        recommendation = "Good convergence - results should be suitable for most purposes"
    elif convergence_rate >= 0.2:
        status = "Fair"
        recommendation = "Some runs converged - consider longer optimization"
    else:
        status = "Poor"
        recommendation = "Poor convergence - extend optimization time or adjust parameters"

    best_overall_objectives = [obj for obj in best_objectives if obj != 'N/A' and obj is not None]
    best_improvement = best_overall_objectives[-1] - best_overall_objectives[0] if len(
        best_overall_objectives) > 1 else 0

    return {
        "Convergence Status": status,
        "Recommendation": recommendation,
        "Converged Runs": f"{converged_runs}/{len(run_data)}",
        "Convergence Rate": f"{convergence_rate:.1%}",
        "Std Deviation": f"{std_final:.6f}",
        "Avg Final Objective": f"{avg_final:.6f}",
        "Avg Improvement": f"{avg_improvement:.6f}",
        "Best Overall Improvement": f"{best_improvement:.6f}",
        "Runtime": f"{runtime:.0f} minutes"
    }
