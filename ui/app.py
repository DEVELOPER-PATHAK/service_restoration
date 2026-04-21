import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import time

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import DistributionNetwork
from core.fitness import RestorationFitness, repair_radiality
from algorithms.bpso import run_bpso
from algorithms.cpso import run_cpso
from algorithms.hybrid_pso_ga import run_hybrid_pso_ga

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Restoration Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .stApp { background: #0a0f1e; color: #e2e8f0; }

  .metric-card {
    background: linear-gradient(135deg, #1a2340 0%, #0d1730 100%);
    border: 1px solid #2d3f6b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    box-shadow: 0 4px 20px rgba(0,120,255,0.1);
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    color: #60a5fa;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
  }
  .status-good  { color: #34d399 !important; }
  .status-warn  { color: #fbbf24 !important; }
  .status-bad   { color: #f87171 !important; }

  .algo-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-bpso   { background: #1e3a6b; color: #93c5fd; border: 1px solid #3b5998; }
  .badge-cpso   { background: #1e4d3a; color: #6ee7b7; border: 1px solid #2d6a50; }
  .badge-hybrid { background: #4d2e1e; color: #fdba74; border: 1px solid #7c4a2a; }

  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #475569;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin: 1rem 0 0.8rem 0;
  }
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
  }
  .stSelectbox > div > div { background: #111827 !important; border-color: #2d3f6b !important; }
  .stSlider > div > div > div { background: #1d4ed8 !important; }
  div[data-testid="stSidebar"] { background: #060d1f !important; border-right: 1px solid #1a2a4a; }
  .stDataFrame { border: 1px solid #2d3f6b; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if 'network' not in st.session_state:
    st.session_state.network = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'comparison' not in st.session_state:
    st.session_state.comparison = []


# ── Helper functions ──────────────────────────────────────────────────────────
def load_default_network():
    # data_path = os.path.join(os.path.dirname(__file__), 'data', 'ieee33.json')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ieee33.json')
    net = DistributionNetwork()
    net.load_from_json(data_path)
    return net


def plot_convergence(histories, labels, colors):
    fig = go.Figure()
    for hist, lbl, col in zip(histories, labels, colors):
        fig.add_trace(go.Scatter(
            y=hist, mode='lines', name=lbl,
            line=dict(color=col, width=2.5),
            hovertemplate='Iter %{x}: %{y:.6f}<extra>' + lbl + '</extra>'
        ))
    fig.update_layout(
        title='Convergence Curves',
        xaxis_title='Iteration',
        yaxis_title='Best Cost',
        plot_bgcolor='#0d1730',
        paper_bgcolor='#0a0f1e',
        font_color='#e2e8f0',
        legend=dict(bgcolor='#111827', bordercolor='#2d3f6b', borderwidth=1),
        xaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
        yaxis=dict(gridcolor='#1e293b', zerolinecolor='#1e293b'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    return fig


def plot_network(network, fault_bus=None, result_states=None):
    """Simple network visualization using node positions."""
    # Define rough positions for IEEE 33-bus
    positions = {
        1: (0, 3), 2: (1, 3), 3: (2, 3), 4: (3, 3), 5: (4, 3),
        6: (5, 3), 7: (6, 3), 8: (7, 3), 9: (8, 3), 10: (9, 3),
        11: (10, 3), 12: (11, 3), 13: (12, 3), 14: (13, 3), 15: (14, 3),
        16: (15, 3), 17: (16, 3), 18: (17, 3),
        19: (1, 2), 20: (2, 2), 21: (3, 2), 22: (4, 2),
        23: (2, 4), 24: (3, 4), 25: (4, 4),
        26: (5, 4), 27: (6, 4), 28: (7, 4), 29: (8, 4), 30: (9, 4),
        31: (10, 4), 32: (11, 4), 33: (12, 4)
    }

    if result_states is not None:
        net = network.apply_switch_states(result_states)
        energized = net.get_energized_buses()\

    else:
        energized = set(network.buses.keys())

    fig = go.Figure()

    # Draw edges
    for bid, br in network.branches.items():
        f, t = br['from'], br['to']
        if f not in positions or t not in positions:
            continue
        x0, y0 = positions[f]
        x1, y1 = positions[t]

        if result_states is not None:
            sw_idx = network.switch_index_map.get(bid)
            if sw_idx is not None:
                is_closed = result_states[sw_idx] > 0.5
            else:
                is_closed = br.get('state', 'closed') == 'closed'
        else:
            is_closed = br.get('state', 'closed') == 'closed'

        if br.get('type', 'auto') == 'manual':
            col = '#f59e0b' if is_closed else '#374151'
            dash = 'dash'
        else:
            col = '#3b82f6' if is_closed else '#1e293b'
            dash = 'solid'

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=col, width=2, dash=dash),
            hoverinfo='none',
            showlegend=False
        ))

    # Draw nodes
    node_x, node_y, node_color, node_text, node_hover = [], [], [], [], []
    for nid, attrs in network.buses.items():
        if nid not in positions:
            continue
        x, y = positions[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(nid))

        if nid == network.substation:
            color = '#f59e0b'
        elif fault_bus and nid == fault_bus:
            color = '#ef4444'
        elif result_states is not None and nid not in energized:
            color = '#374151'
        else:
            p = attrs.get('priority', 3)
            colors_map = {1: '#34d399', 2: '#60a5fa', 3: '#a78bfa'}
            color = colors_map.get(p, '#a78bfa')

        node_color.append(color)
        node_hover.append(
            f"Bus {nid}<br>Load: {attrs.get('load_kw', 0)} kW<br>"
            f"Priority: {attrs.get('priority', 3)}<br>"
            f"Status: {'Energized' if nid in energized else 'Out-of-service'}"
        )

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(color=node_color, size=18, line=dict(color='#0a0f1e', width=1.5)),
        text=node_text,
        textfont=dict(size=8, color='white'),
        textposition='middle center',
        hovertext=node_hover,
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor='#0d1730',
        paper_bgcolor='#0a0f1e',
        font_color='#e2e8f0',
        margin=dict(l=5, r=5, t=5, b=5),
        height=280,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def metric_card(label, value, unit='', status='normal'):
    cls = {'normal': '', 'good': 'status-good', 'warn': 'status-warn', 'bad': 'status-bad'}
    c = cls.get(status, '')
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {c}">{value}<span style="font-size:0.9rem;color:#64748b;"> {unit}</span></div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# ⚡ RestoreOpt")
    st.markdown("<p style='color:#475569;font-size:0.8rem;'>Supply Restoration via Metaheuristics</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Network selection
    st.markdown('<div class="section-header">Network</div>', unsafe_allow_html=True)
    network_choice = st.selectbox("Test System", ["IEEE 33-Bus (Default)", "Upload Custom JSON"])

    if network_choice == "IEEE 33-Bus (Default)":
        if st.button("Load IEEE 33-Bus"):
            st.session_state.network = load_default_network()
            st.session_state.results = {}
            st.session_state.comparison = []
            st.success("IEEE 33-Bus loaded!")
    else:
        uploaded = st.file_uploader("Upload network JSON", type=['json'])
        if uploaded:
            try:
                data = json.load(uploaded)
                net = DistributionNetwork()
                net.load_from_dict(data)
                st.session_state.network = net
                st.session_state.results = {}
                st.success(f"Loaded: {net.name}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Fault injection
    st.markdown('<div class="section-header">Fault Injection</div>', unsafe_allow_html=True)
    net = st.session_state.get('network')
    if net:
        bus_ids = sorted([b for b in net.buses.keys() if b != net.substation])
        fault_bus = st.selectbox("Fault at Bus", bus_ids, index=min(25, len(bus_ids)-1))
    else:
        fault_bus = None
        st.info("Load a network first.")

    st.markdown("---")

    # Algorithm settings
    st.markdown('<div class="section-header">Algorithm</div>', unsafe_allow_html=True)
    algo = st.selectbox("Algorithm", ["BPSO", "Continuous PSO", "Hybrid PSO-GA", "Compare All Three"])
    n_particles = st.slider("Particles", 10, 80, 30)
    max_iter = st.slider("Iterations", 10, 150, 50)
    w_inertia = st.slider("Inertia (w)", 0.1, 1.0, 0.7)
    c1 = st.slider("c1 (personal)", 0.5, 3.0, 1.5)
    c2 = st.slider("c2 (global)", 0.5, 3.0, 1.5)

    st.markdown("---")
    st.markdown('<div class="section-header">Objective Weights (A,F,K,L,M)</div>', unsafe_allow_html=True)
    w_ens      = st.slider("A: Energy Not Supplied",  0.0, 1.0, 0.5)
    w_priority = st.slider("F: Load Priority",        0.0, 1.0, 0.2)
    w_seq      = st.slider("K: Switch Sequence",      0.0, 1.0, 0.1)
    w_num      = st.slider("L: Number of Switches",   0.0, 1.0, 0.1)
    w_type     = st.slider("M: Switch Type",          0.0, 1.0, 0.1)

    weights = {'ens': w_ens, 'priority': w_priority,
               'seq': w_seq, 'num_sw': w_num, 'type': w_type}

    st.markdown("---")
    run_btn = st.button("🚀  Run Optimization", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Supply Restoration Optimizer")
st.markdown("<p style='color:#64748b;margin-top:-1rem;'>Smart Distribution System · Self-Healing via PSO/GA</p>",
            unsafe_allow_html=True)

if st.session_state.network is None:
    st.warning("👈 Load a network from the sidebar to begin.")
    st.stop()

net = st.session_state.network

# ── Network info ──────────────────────────────────────────────────────────────
tab_main, tab_compare, tab_switches, tab_about = st.tabs(
    ["🔬 Optimization", "📊 Comparison", "🔌 Switch Table", "📚 About"]
)

with tab_main:
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.markdown(metric_card("Total Buses", net.G.number_of_nodes()), unsafe_allow_html=True)
    with col_info2:
        st.markdown(metric_card("Total Switches", net.get_num_switches()), unsafe_allow_html=True)
    with col_info3:
        st.markdown(metric_card("Total Load", f"{net.get_total_load():.0f}", "kW"), unsafe_allow_html=True)
    with col_info4:
        fb_label = f"Bus {fault_bus}" if fault_bus else "None"
        st.markdown(metric_card("Fault Bus", fb_label, status='bad' if fault_bus else 'normal'),
                    unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### Network Topology")
    fig_net = plot_network(net, fault_bus=fault_bus)
    st.plotly_chart(fig_net, use_container_width=True)

    # Legend
    st.markdown("""
    <div style='display:flex;gap:1.5rem;font-size:0.75rem;color:#64748b;margin-top:-0.5rem;'>
      <span>🟡 Substation</span>
      <span>🔴 Fault Bus</span>
      <span>🟢 Priority 1</span>
      <span>🔵 Priority 2</span>
      <span>🟣 Priority 3</span>
      <span style='color:#374151'>⚫ Out-of-service</span>
    </div>""", unsafe_allow_html=True)

    # ── Run optimization ──────────────────────────────────────────────────────
    if run_btn and fault_bus:
        algos_to_run = []
        if algo == "Compare All Three":
            algos_to_run = ["BPSO", "Continuous PSO", "Hybrid PSO-GA"]
        else:
            algos_to_run = [algo]

        runner_map = {
            "BPSO": run_bpso,
            "Continuous PSO": run_cpso,
            "Hybrid PSO-GA": run_hybrid_pso_ga
        }

        st.markdown("---")
        progress_bar = st.progress(0)
        status_txt = st.empty()

        st.session_state.comparison = []

        for a_idx, a_name in enumerate(algos_to_run):
            status_txt.markdown(f"**Running {a_name}...**")
            runner = runner_map[a_name]

            prog_placeholder = st.empty()

            def make_callback(placeholder, a_idx_=a_idx, total=len(algos_to_run), a_name_=a_name):
                def cb(it, max_it, cost):
                    base = a_idx_ / total
                    frac = base + (it / max_it) / total
                    progress_bar.progress(min(frac, 1.0))
                    placeholder.markdown(f"*{a_name_}: iter {it}/{max_it} — cost {cost:.6f}*")
                return cb

            best_pos, best_cost, history, metrics = runner(
                net,
                fault_bus=fault_bus,
                weights=weights,
                n_particles=n_particles,
                max_iter=max_iter,
                w=w_inertia, c1=c1, c2=c2,
                callback=make_callback(prog_placeholder)
            )

            prog_placeholder.empty()
            st.session_state.results[a_name] = {
                'pos': best_pos,
                'cost': best_cost,
                'history': history,
                'metrics': metrics
            }
            st.session_state.comparison.append(metrics)

        progress_bar.progress(1.0)
        status_txt.markdown("✅ **Optimization complete!**")

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.results:
        st.markdown("---")
        st.markdown("#### Results")

        # Convergence plot
        histories, labels, colors = [], [], []
        color_map = {"BPSO": "#60a5fa", "Continuous PSO": "#34d399", "Hybrid PSO-GA": "#fb923c"}
        for a_name, res in st.session_state.results.items():
            histories.append(res['history'])
            labels.append(a_name)
            colors.append(color_map.get(a_name, '#a78bfa'))

        # st.plotly_chart(plot_convergence(histories, labels, colors), use_container_width=True)

        # Per-algorithm detailed result
        for a_name, res in st.session_state.results.items():
            m = res['metrics']
            badge_map = {
                "BPSO": "badge-bpso",
                "Continuous PSO": "badge-cpso",
                "Hybrid PSO-GA": "badge-hybrid"
            }
            badge = badge_map.get(a_name, "badge-bpso")
            st.markdown(f'<span class="algo-badge {badge}">{a_name}</span>', unsafe_allow_html=True)

            cc = st.columns(6)
            with cc[0]:
                eff = m['restoration_efficiency']
                s = 'good' if eff >= 95 else ('warn' if eff >= 80 else 'bad')
                st.markdown(metric_card("Efficiency", f"{eff:.1f}", "%", s), unsafe_allow_html=True)
            with cc[1]:
                st.markdown(metric_card("ENS", f"{m['ens_kw']:.0f}", "kW"), unsafe_allow_html=True)
            with cc[2]:
                st.markdown(metric_card("Switches Operated", m['num_switches_operated']), unsafe_allow_html=True)
            with cc[3]:
                st.markdown(metric_card("Manual Ops", m['manual_switches_operated']), unsafe_allow_html=True)
            with cc[4]:
                r_s = 'good' if m['is_radial'] else 'bad'
                st.markdown(metric_card("Radial", "✓" if m['is_radial'] else "✗", status=r_s), unsafe_allow_html=True)
            with cc[5]:
                st.markdown(metric_card("Runtime", f"{m['runtime_s']}", "s"), unsafe_allow_html=True)

            # Post-restoration network view
            with st.expander(f"Post-restoration network — {a_name}"):
                fig_after = plot_network(net, fault_bus=fault_bus, result_states=res['pos'])
                # st.plotly_chart(fig_after, use_container_width=True)
                st.plotly_chart(fig_after, use_container_width=True, key=f"net_after_{a_name}")
            st.markdown("")

    elif run_btn and not fault_bus:
        st.warning("Select a fault bus from the sidebar first.")


with tab_compare:
    st.markdown("#### Algorithm Comparison Table")
    if st.session_state.comparison:
        df = pd.DataFrame(st.session_state.comparison)
        display_cols = ['algorithm', 'restoration_efficiency', 'ens_kw', 'restored_load_kw',
                        'num_switches_operated', 'manual_switches_operated',
                        'is_radial', 'convergence_iter', 'runtime_s', 'total_cost']
        df_show = df[[c for c in display_cols if c in df.columns]].copy()
        df_show.columns = [c.replace('_', ' ').title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True)

        # Bar chart comparison
        if len(st.session_state.comparison) > 1:
            fig_bar = go.Figure()
            algos = [m['algorithm'] for m in st.session_state.comparison]
            effs = [m['restoration_efficiency'] for m in st.session_state.comparison]
            fig_bar.add_trace(go.Bar(x=algos, y=effs,
                                     marker_color=['#60a5fa', '#34d399', '#fb923c'],
                                     text=[f"{e:.1f}%" for e in effs],
                                     textposition='outside'))
            fig_bar.update_layout(
                title='Restoration Efficiency Comparison (%)',
                yaxis=dict(range=[0, 105], gridcolor='#1e293b'),
                plot_bgcolor='#0d1730', paper_bgcolor='#0a0f1e',
                font_color='#e2e8f0',
                margin=dict(l=10, r=10, t=40, b=10),
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Radar chart
            cats = ['Efficiency', 'Speed', 'Min Switches', 'Radial']
            fig_radar = go.Figure()
            col_map = {"BPSO": "#60a5fa", "Continuous PSO": "#34d399", "Hybrid PSO-GA": "#fb923c"}
            for m in st.session_state.comparison:
                max_iter_val = max_iter if max_iter > 0 else 1
                max_sw = net.get_num_switches() if net.get_num_switches() > 0 else 1
                vals = [
                    m['restoration_efficiency'] / 100,
                    1 - m['runtime_s'] / max(m['runtime_s'] + 0.001, 5),
                    1 - m['num_switches_operated'] / max_sw,
                    1.0 if m['is_radial'] else 0.0
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill='toself',
                    name=m['algorithm'],
                    line_color=col_map.get(m['algorithm'], '#a78bfa'),
                    fillcolor=col_map.get(m['algorithm'], '#a78bfa'),
                    opacity=0.25
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e293b', tickfont_color='#64748b'),
                    angularaxis=dict(gridcolor='#1e293b', tickfont_color='#94a3b8'),
                    bgcolor='#0d1730'
                ),
                paper_bgcolor='#0a0f1e',
                font_color='#e2e8f0',
                title='Performance Radar',
                height=350,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Run an optimization first (sidebar → Compare All Three for best results).")


with tab_switches:
    st.markdown("#### Switch State Table")
    if st.session_state.results and net:
        switch_info = net.get_switch_info()
        orig_states = net.get_switch_states()

        for a_name, res in st.session_state.results.items():
            st.markdown(f'<span class="algo-badge badge-bpso">{a_name}</span>', unsafe_allow_html=True)
            rows = []
            for idx, info in enumerate(switch_info):
                orig = "Closed" if orig_states[idx] > 0.5 else "Open"
                new_s = "Closed" if res['pos'][idx] > 0.5 else "Open"
                changed = orig != new_s
                rows.append({
                    'Switch ID': info['id'],
                    'From': info['from'],
                    'To': info['to'],
                    'Type': info.get('type', 'auto').upper(),
                    'Pre-Fault': orig,
                    'Post-Restoration': new_s,
                    'Operated': '✓' if changed else ''
                })
            df_sw = pd.DataFrame(rows)
            st.dataframe(df_sw, use_container_width=True)
            st.markdown("")
    else:
        st.info("Run optimization to see switch states.")


with tab_about:
    st.markdown("""
    ## About This Tool

    This application implements **Binary PSO (BPSO)**, **Continuous PSO**, and a **Hybrid PSO-GA** algorithm
    for electric supply restoration in smart distribution systems.

    It is based on the research framework described in:
    > *Goda et al., "Electric supply restoration in self-healed smart distribution systems: a review,"
    > Energy Informatics (2025) 8:114*

    ### Challenges Addressed
    | Code | Challenge | Description |
    |------|-----------|-------------|
    | A | Energy Not Supplied | Minimize ENS after fault |
    | F | Load Priority | Serve high-priority loads first |
    | K | Switch Sequence | Generate optimal switch command order |
    | L | Number of Switches | Minimize total switch operations |
    | M | Switch Type | Prefer automatic over manual switches |

    ### Algorithms
    - **BPSO**: Sigmoid-based binary position update, directly models open/closed switch states
    - **Continuous PSO**: Real-valued positions thresholded to binary; smoother velocity landscape
    - **Hybrid PSO-GA**: PSO with crossover/mutation on top particles to escape local optima

    ### Constraint Handling
    - **Radiality** is enforced after each position update via a repair operator
    - **Fault isolation** forces switches around the faulted bus to open
    - **Radial violations** add a large penalty to the fitness function

    ### How to Use
    1. Load IEEE 33-Bus from the sidebar
    2. Select a fault bus
    3. Tune weights for challenges A, F, K, L, M
    4. Choose algorithm and run
    5. View convergence, metrics, and switch states

    ---
    *Developed for MANIT Bhopal · EE Dept*
    """)
