"""
streamlit_app.py
Minimal monochrome Streamlit dashboard for SLATE.
Provides tabular inspection, eigenvalue complex plane, 2D phase portrait,
and linear algebraic decomposition.
No emojis. No em dashes.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.pipeline import SLATEPipeline

st.set_page_config(
    page_title="SLATE :: spectral dynamics lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal Monochrome Styling
st.markdown("""
<style>
    .reportview-container, .main, body { background-color: #000000; color: #ffffff; }
    h1, h2, h3, h4 { font-family: monospace; color: #ffffff; }
    .stMetric { background: #0a0a0a; padding: 10px; border: 1px solid #222222; }
</style>
""", unsafe_allow_html=True)

st.title("[ SLATE ] :: spectral analysis of neural network dynamics")
st.caption("Discrete dynamical systems, orthonormal projection, and spectral stability in state space")

# Sidebar Controls
st.sidebar.header(":: parameters ::")
hidden_dim = st.sidebar.selectbox("latent dimension n:", [4, 8, 12, 16], index=1)
recurrent_scale = st.sidebar.slider("recurrent scale (stability test):", min_value=0.6, max_value=1.5, value=1.0, step=0.05)

@st.cache_data(show_spinner="computing mathematical pipeline...")
def get_pipeline_data(h_dim, scale):
    p = SLATEPipeline(hidden_dim=h_dim)
    return p.run_experiment(recurrent_scale=scale)

results = get_pipeline_data(hidden_dim, recurrent_scale)

# Telemetry Metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("timesteps (T)", results["meta"]["timesteps_T"])
col2.metric("latent dimension (n)", results["meta"]["hidden_dim_n"])
col3.metric("spectral radius rho(A)", f"{results['spectral']['spectral_radius']:.3f}")
col4.metric("surrogate r^2 fit", f"{results['dynamics']['r2_fit_score']:.3f}")
col5.metric("stability", results["spectral"]["stability_class"].split(' ')[0])

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "[ 01 / trajectory ]", 
    "[ 02 / spectrum ]", 
    "[ 03 / projection ]",
    "[ 04 / surrogate ]", 
    "[ 05 / rref ]", 
    "[ 06 / prediction ]"
])

with tab1:
    st.subheader("orthonormal subspace phase portrait")
    coords_2d = np.array(results["projection"]["coords_2d"])
    surrogate_2d = np.array(results["dynamics"]["coords_surrogate_2d"])
    
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#000000")
    ax.set_facecolor("#000000")
    
    # Plot surrogate path (dashed gray)
    ax.plot(surrogate_2d[:, 0], surrogate_2d[:, 1], '--', color="#666666", alpha=0.7, label="linear surrogate (A h_t)")
    
    # Plot actual trajectory (solid white)
    ax.plot(coords_2d[:, 0], coords_2d[:, 1], color="#ffffff", lw=1.2, label="actual rnn trajectory")
    ax.scatter(coords_2d[::3, 0], coords_2d[::3, 1], color="#888888", s=10)
    
    ax.set_xlabel("basis vector q_1", color="#888888", fontfamily="monospace")
    ax.set_ylabel("basis vector q_2", color="#888888", fontfamily="monospace")
    ax.tick_params(colors="#888888")
    ax.grid(True, color="#1a1a1a", ls=":")
    ax.legend(facecolor="#000000", edgecolor="#333333", labelcolor="#ffffff")
    st.pyplot(fig)

with tab2:
    st.subheader("spectral analysis and complex unit circle")
    st.write(f"status: {results['spectral']['stability_desc']}")
    
    modes = results["spectral"]["modes"]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="#000000")
    ax.set_facecolor("#000000")
    
    # Unit Circle |lambda| = 1
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '--', color="#ffffff", lw=1, label="unit circle |lambda| = 1")
    
    # Spectral Radius circle
    rho = results["spectral"]["spectral_radius"]
    ax.plot(rho * np.cos(theta), rho * np.sin(theta), ':', color="#666666", label=f"rho(A) = {rho:.3f}")
    
    # Plot Eigenvalues
    for m in modes:
        ax.scatter([m["real"]], [m["imag"]], color="#ffffff", s=60, zorder=5)
        ax.plot([0, m["real"]], [0, m["imag"]], color="#444444", lw=1)
        ax.text(m["real"] + 0.03, m["imag"] + 0.03, f"lambda_{m['index']}", color="#888888", fontsize=8, fontfamily="monospace")
        
    ax.axhline(0, color="#222222", lw=1)
    ax.axvline(0, color="#222222", lw=1)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("real axis re(lambda)", color="#888888", fontfamily="monospace")
    ax.set_ylabel("imag axis im(lambda)", color="#888888", fontfamily="monospace")
    ax.tick_params(colors="#888888")
    ax.legend(facecolor="#000000", edgecolor="#333333", labelcolor="#ffffff")
    st.pyplot(fig)
    
    df_modes = pd.DataFrame(modes)[["index", "real", "imag", "magnitude", "angle_deg", "period_steps", "mode_type"]]
    st.dataframe(df_modes, use_container_width=True)

with tab3:
    st.subheader("basis and orthogonal projection")
    st.markdown(r"""
    The centered hidden states $H \in \mathbb{R}^{T \times n}$ are projected using QR decomposition:
    $$H_c^T = Q R \implies B = Q_{:,:k}$$
    $$z_t = B^T (h_t - \mu)$$
    """)
    st.write(f"variance retained in 2d: `{results['projection']['variance_retained_2d']}%`")
    st.write(f"variance retained in 3d: `{results['projection']['variance_retained_3d']}%`")
    st.write(f"orthonormality verification error: `{results['projection']['orthonormality_error']:.2e}`")

with tab4:
    st.subheader("least squares linear surrogate dynamics")
    st.markdown(r"""
    Solving for transition operator $A \in \mathbb{R}^{n \times n}$:
    $$h_{t+1} \approx A h_t \implies A = (H_{\text{prev}}^+ H_{\text{next}})^T$$
    """)
    st.write(f"mean one-step residual: `{results['dynamics']['mean_step_error']}`")
    st.write(f"coefficient of determination r^2: `{results['dynamics']['r2_fit_score']}`")
    st.line_chart(results["dynamics"]["step_errors"])

with tab5:
    st.subheader("matrix structure and rref (sympy)")
    rref = results["rref"]
    c1, c2 = st.columns(2)
    with c1:
        st.write("numerical rank:", rref["rank"])
        st.write("pivot dimensions:", rref["pivot_dimensions"])
        st.write("redundant dimensions:", rref["redundant_dimensions"])
        st.write("redundancy percentage:", f"{rref['redundancy_pct']}%")
    with c2:
        st.write("transition matrix A:")
        st.dataframe(pd.DataFrame(results["matrices"]["A_transition"]), use_container_width=True)

with tab6:
    st.subheader("time-series prediction")
    preds = results["predictions"]
    st.write(f"rmse: `{preds['rmse']}` | mae: `{preds['mae']}` | r^2: `{preds['r2']}`")
    df_eval = pd.DataFrame({
        "actual": preds["actual_y"][:168],
        "predicted": preds["predicted_y"][:168]
    })
    st.line_chart(df_eval)
