import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="UHPC-Strengthened RC Beam – Moment Gain Predictor",
    layout="wide"
)

# ============================================================
# LOAD SAVED MODEL (PIPELINE)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("xgboost_best_only.pkl")

model = load_model()

# ============================================================
# BLUE RANGE TEXT STYLE (for ranges shown under each input)
# ============================================================
BLUE = "#1f77b4"  # blue text (safe default)

def range_text(s: str):
    st.markdown(f"<div style='color:{BLUE}; font-size:13px; margin-top:-10px;'>{s}</div>",
                unsafe_allow_html=True)

# ============================================================
# INPUT SPECIFICATION (ORDER MUST MATCH TRAINING FEATURES)
# ============================================================
numerical_cols = [
    "L (mm)", "a", "bw (mm)", "d (mm)", "fc_rc (MPa)", "ρsl_rc (%)",
    "fy_rc (MPa)", "fyv_rc (MPa)", "ρv_rc (%)", "t_uhpc (mm)",
    "E_uhpc (GPa)", "ft_uhpc (MPa)", "fc_uhpc (MPa)",
    "ρ_uhpc (%)", "vf (%)", "λs"
]
categorical_cols = ["layout", "iface"]

# Ranges for UI (min, max, default)
RANGES = {
    "L (mm)": (760, 3300, 1500),
    "a": (200, 1350, 600),
    "bw (mm)": (100, 250, 150),
    "d (mm)": (110, 425, 250),

    "fc_rc (MPa)": (20.1, 70.07, 40.0),
    "ρsl_rc (%)": (0.0, 8.31, 1.5),
    "fy_rc (MPa)": (0.0, 600.0, 420.0),
    "fyv_rc (MPa)": (0.0, 610.0, 420.0),
    "ρv_rc (%)": (0.0, 1.43, 0.5),

    "t_uhpc (mm)": (5.0, 70.0, 30.0),
    "E_uhpc (GPa)": (34.6, 145.0, 50.0),
    "ft_uhpc (MPa)": (5.0, 16.0, 10.0),
    "fc_uhpc (MPa)": (102.2, 204.0, 150.0),
    "ρ_uhpc (%)": (0.0, 3.35, 1.0),
    "vf (%)": (0.5, 3.0, 2.0),
    "λs": (26.79, 125.0, 65.0),
}

LAYOUT_OPTIONS = ["1-sided", "2-sided", "3-sided", "C-sided", "T-sided"]
IFACE_OPTIONS = [
    "Roughned & UHPC casting",
    "Pre-cast UHPC & Anchorage",
    "Roughned, Epoxy adhesive & UHPC casting"
]

# ============================================================
# APP TITLE
# ============================================================
st.title("UHPC-Strengthened RC Beam – Percentage Moment Gain (M%) Predictor")
st.write("Enter beam, material, UHPC strengthening, and interface parameters to predict **M(%)**.")

st.divider()

# ============================================================
# TWO EQUAL COLUMNS LAYOUT (EQUAL ROWS)
# We'll split the 18 inputs into 9 left + 9 right (exactly)
# ============================================================
left, right = st.columns(2, gap="large")

# ---- Define left column fields (9) ----
left_fields = [
    ("RC beam geometrical parameters", "L (mm)", "Span length"),
    ("RC beam geometrical parameters", "a", "Shear span ratio"),
    ("RC beam geometrical parameters", "bw (mm)", "Beam width"),
    ("RC beam geometrical parameters", "d (mm)", "Effective depth"),

    ("RC material properties", "fc_rc (MPa)", "Concrete compressive strength"),
    ("RC material properties", "ρsl_rc (%)", "Longitudinal reinforcement ratio"),
    ("RC material properties", "fy_rc (MPa)", "Longitudinal steel yield strength"),
    ("RC material properties", "fyv_rc (MPa)", "Shear reinforcement yield strength"),
    ("RC material properties", "ρv_rc (%)", "Shear reinforcement ratio"),
]

# ---- Define right column fields (9) ----
right_fields = [
    ("UHPC strengthening parameters", "t_uhpc (mm)", "UHPC layer thickness"),
    ("UHPC strengthening parameters", "E_uhpc (GPa)", "UHPC elastic modulus"),
    ("UHPC strengthening parameters", "ft_uhpc (MPa)", "UHPC tensile strength"),
    ("UHPC strengthening parameters", "fc_uhpc (MPa)", "UHPC compressive strength"),
    ("UHPC strengthening parameters", "ρ_uhpc (%)", "UHPC reinforcement ratio"),
    ("UHPC strengthening parameters", "vf (%)", "Fiber volume fraction"),
    ("UHPC strengthening parameters", "λs", "Steel fiber aspect ratio"),

    ("Interface and configuration parameters", "layout", "UHPC layout"),
    ("Interface and configuration parameters", "iface", "RC & UHPC interface preparation"),
]

# Storage for user inputs
inputs = {}

# ============================================================
# RENDER INPUTS – LEFT COLUMN
# ============================================================
with left:
    current_group = None
    for group, colname, label in left_fields:
        if group != current_group:
            st.subheader(group)
            current_group = group

        vmin, vmax, vdefault = RANGES[colname]

        # slider is intuitive for ranges; numeric input can be added if needed
        step = 1.0
        if any(x in colname for x in ["ρ", "fc_", "ft_", "E_", "vf", "λs"]):
            step = 0.01

        val = st.slider(
            label=label,
            min_value=float(vmin),
            max_value=float(vmax),
            value=float(vdefault),
            step=float(step),
            key=f"slider_{colname}"
        )
        range_text(f"Range: {vmin} – {vmax}")
        inputs[colname] = val

# ============================================================
# RENDER INPUTS – RIGHT COLUMN
# ============================================================
with right:
    current_group = None
    for group, colname, label in right_fields:
        if group != current_group:
            st.subheader(group)
            current_group = group

        if colname in RANGES:
            vmin, vmax, vdefault = RANGES[colname]
            step = 1.0
            if any(x in colname for x in ["ρ", "fc_", "ft_", "E_", "vf", "λs"]):
                step = 0.01

            val = st.slider(
                label=label,
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(vdefault),
                step=float(step),
                key=f"slider_{colname}"
            )
            range_text(f"Range: {vmin} – {vmax}")
            inputs[colname] = val

        elif colname == "layout":
            val = st.selectbox(label=label, options=LAYOUT_OPTIONS, index=2)
            range_text("Options: 1-sided, 2-sided, 3-sided, C-sided, T-sided")
            inputs[colname] = val

        elif colname == "iface":
            val = st.selectbox(label=label, options=IFACE_OPTIONS, index=0)
            range_text("Options: Roughened casting; Pre-cast & anchorage; Roughened+epoxy+casting")
            inputs[colname] = val

st.divider()

# ============================================================
# PREDICTION
# ============================================================
col_predict, col_out = st.columns([1, 2], gap="large")

with col_predict:
    st.subheader("Prediction")
    predict_btn = st.button("Predict M(%)", type="primary", use_container_width=True)

with col_out:
    st.subheader("Output")

    if predict_btn:
        # Build input dataframe in correct column order
        row = {**{c: inputs[c] for c in numerical_cols},
               **{c: inputs[c] for c in categorical_cols}}

        X_input = pd.DataFrame([row], columns=numerical_cols + categorical_cols)

        # Predict
        y_hat = float(model.predict(X_input)[0])
        y_hat = max(y_hat, 0.0)  # enforce non-negative

        st.success(f"Predicted Percentage Moment Gain, M(%) = {y_hat:.3f}")

        # Optional: show the input row
        with st.expander("Show input values"):
            st.dataframe(X_input, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.caption("Model: XGBoost (Optuna-tuned) pipeline with preprocessing + one-hot encoding.")
