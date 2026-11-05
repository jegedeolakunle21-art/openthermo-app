
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
from pathlib import Path

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="OpenThermoAI | Thermal Performance",
    page_icon="🌡️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "openthermoai_model.joblib"
TEMP_MODEL_PATH = MODEL_DIR / "openthermoai_model.download"

METRICS_PATH = BASE / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE / "data" / "optimization_scenarios.csv"

GOOGLE_DRIVE_FILE_ID = "18ZPMFhU303wK_-9Hz-gn4aSwCr-IGHRj"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. MODEL FEATURES AND TARGETS
# =====================================================

FEATURE_LABELS = {
    "Th_in_C": "Hot-fluid inlet temperature (°C)",
    "Tc_in_C": "Cold-fluid inlet temperature (°C)",
    "mdot_h_kg_s": "Hot-fluid mass flow rate (kg/s)",
    "mdot_c_kg_s": "Cold-fluid mass flow rate (kg/s)",
    "area_m2": "Heat-transfer area (m²)",
    "U_W_m2K": "Overall heat-transfer coefficient (W/m²K)",
    "Dh_h_m": "Hot-side hydraulic diameter (m)",
    "Dh_c_m": "Cold-side hydraulic diameter (m)",
    "L_h_m": "Hot-side length (m)",
    "L_c_m": "Cold-side length (m)"
}

TARGET_LABELS = {
    "Q_W": "Heat-transfer rate (W)",
    "Th_out_C": "Hot-fluid outlet temperature (°C)",
    "Tc_out_C": "Cold-fluid outlet temperature (°C)",
    "dp_h_Pa": "Hot-side pressure drop (Pa)",
    "dp_c_Pa": "Cold-side pressure drop (Pa)",
    "Nu_h": "Hot-side Nusselt number",
    "Nu_c": "Cold-side Nusselt number",
    "h_h_W_m2K": "Hot-side heat-transfer coefficient (W/m²K)",
    "h_c_W_m2K": "Cold-side heat-transfer coefficient (W/m²K)"
}

POSITIVE_FEATURES = [
    "mdot_h_kg_s",
    "mdot_c_kg_s",
    "area_m2",
    "U_W_m2K",
    "Dh_h_m",
    "Dh_c_m",
    "L_h_m",
    "L_c_m"
]

# =====================================================
# 3. LOAD EXISTING TRAINED MODEL
# =====================================================

@st.cache_resource(show_spinner=False)
def load_bundle():

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()

        downloaded = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(TEMP_MODEL_PATH),
            quiet=False
        )

        if (
            downloaded is None
            or not TEMP_MODEL_PATH.exists()
            or TEMP_MODEL_PATH.stat().st_size < 1_000_000
        ):
            raise FileNotFoundError(
                "Could not retrieve the existing trained model."
            )

        TEMP_MODEL_PATH.replace(MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError("The model bundle must be a dictionary.")

    required_keys = ["model", "features", "targets"]

    missing = [
        key for key in required_keys
        if key not in bundle
    ]

    if missing:
        raise KeyError(f"Missing model components: {missing}")

    return bundle


try:
    with st.spinner("Loading OpenThermoAI trained model..."):
        bundle = load_bundle()

    model = bundle["model"]
    FEATURES = list(bundle["features"])
    TARGETS = list(bundle["targets"])

except Exception as error:
    st.error("OpenThermoAI could not load its trained model.")
    st.exception(error)
    st.stop()


# =====================================================
# 4. VERIFY MODEL CONFIGURATION
# =====================================================

if len(FEATURES) != 10:
    st.error(f"Expected 10 features, but found {len(FEATURES)}.")
    st.stop()

if len(TARGETS) != 9:
    st.error(f"Expected 9 targets, but found {len(TARGETS)}.")
    st.stop()

if not hasattr(model, "predict"):
    st.error("The loaded model does not have a predict method.")
    st.stop()

if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
    st.error("The model's expected input dimension does not match.")
    st.stop()


# =====================================================
# 5. PREDICTION FUNCTION
# =====================================================

def make_predictions(input_data):

    # Always preserve the exact feature order
    input_df = pd.DataFrame(
        input_data,
        columns=FEATURES
    )

    predictions = model.predict(input_df)

    predictions = np.asarray(predictions)

    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    if predictions.shape[1] != len(TARGETS):
        raise ValueError(
            f"Model returned {predictions.shape[1]} outputs; "
            f"expected {len(TARGETS)}."
        )

    return pd.DataFrame(
        predictions,
        columns=TARGETS
    )


# =====================================================
# 6. APPLICATION HEADER
# =====================================================

st.title("🌡️ OpenThermoAI")

st.subheader(
    "Machine Learning-Based Heat Exchanger "
    "Performance Prediction and Analysis"
)

st.markdown(
    """
    OpenThermoAI uses a trained Extra Trees regression model
    to predict thermal and hydraulic performance parameters
    of heat exchanger systems.
    """
)

model_size = MODEL_PATH.stat().st_size / (1024 * 1024)

st.success("Existing OpenThermoAI model loaded successfully.")

st.caption(
    f"Model: ExtraTreesRegressor | "
    f"Size: {model_size:.2f} MB | "
    f"Inputs: {len(FEATURES)} | "
    f"Outputs: {len(TARGETS)}"
)

st.divider()


# =====================================================
# 7. NAVIGATION
# =====================================================

page = st.sidebar.radio(
    "APPLICATION NAVIGATION",
    [
        "Home",
        "Performance Prediction",
        "Scenario Analysis",
        "About OpenThermoAI"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model status: Ready")

st.sidebar.caption(
    "OpenThermoAI | Thermal Systems Intelligence"
)


# =====================================================
# 8. HOME DASHBOARD
# =====================================================

if page == "Home":

    st.header("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "Extra Trees")
    col2.metric("Input Features", len(FEATURES))
    col3.metric("Prediction Outputs", len(TARGETS))

    st.divider()

    st.subheader("Available Capabilities")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            **Performance Prediction**

            - Heat-transfer rate
            - Hot-fluid outlet temperature
            - Cold-fluid outlet temperature
            - Pressure drops
            - Nusselt numbers
            - Convective heat-transfer coefficients
            """
        )

    with c2:
        st.markdown(
            """
            **Engineering Analysis**

            - Batch scenario evaluation
            - Model validation metrics
            - Prediction result exports
            - Thermal performance assessment
            - Model configuration inspection
            """
        )

    st.divider()

    st.subheader("Model Input Configuration")

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            FEATURE_LABELS.get(x, x)
            for x in FEATURES
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 9. PERFORMANCE PREDICTION
# =====================================================

elif page == "Performance Prediction":

    st.header("Heat Exchanger Performance Prediction")

    st.write(
        "Enter the operating and geometric parameters "
        "of your heat exchanger."
    )

    st.info(
        "Enter actual operating or design values. "
        "The zero values below are placeholders, not recommended "
        "operating conditions. The model's training ranges have "
        "not yet been established."
    )

    with st.form("prediction_form"):

        st.subheader("Thermal Operating Conditions")

        col1, col2 = st.columns(2)

        with col1:
            Th_in_C = st.number_input(
                "Hot-fluid inlet temperature (°C)",
                value=0.0,
                format="%.4f"
            )

            mdot_h_kg_s = st.number_input(
                "Hot-fluid mass flow rate (kg/s)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

            area_m2 = st.number_input(
                "Heat-transfer area (m²)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

            Dh_h_m = st.number_input(
                "Hot-side hydraulic diameter (m)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

            L_h_m = st.number_input(
                "Hot-side length (m)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

        with col2:
            Tc_in_C = st.number_input(
                "Cold-fluid inlet temperature (°C)",
                value=0.0,
                format="%.4f"
            )

            mdot_c_kg_s = st.number_input(
                "Cold-fluid mass flow rate (kg/s)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

            U_W_m2K = st.number_input(
                "Overall heat-transfer coefficient (W/m²K)",
                value=0.0,
                min_value=0.0,
                format="%.4f"
            )

            Dh_c_m = st.number_input(
                "Cold-side hydraulic diameter (m)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

            L_c_m = st.number_input(
                "Cold-side length (m)",
                value=0.0,
                min_value=0.0,
                format="%.5f"
            )

        submitted = st.form_submit_button(
            "Predict Thermal Performance",
            type="primary",
            use_container_width=True
        )

    if submitted:

        values = {
            "Th_in_C": Th_in_C,
            "Tc_in_C": Tc_in_C,
            "mdot_h_kg_s": mdot_h_kg_s,
            "mdot_c_kg_s": mdot_c_kg_s,
            "area_m2": area_m2,
            "U_W_m2K": U_W_m2K,
            "Dh_h_m": Dh_h_m,
            "Dh_c_m": Dh_c_m,
            "L_h_m": L_h_m,
            "L_c_m": L_c_m
        }

        invalid = [
            name for name in POSITIVE_FEATURES
            if values[name] <= 0
        ]

        if invalid:

            st.error(
                "Please enter positive values for: "
                + ", ".join(invalid)
            )

        elif not all(np.isfinite(v) for v in values.values()):

            st.error("All input values must be finite numbers.")

        else:

            try:
                input_row = [
                    [values[feature] for feature in FEATURES]
                ]

                result = make_predictions(input_row)

                st.session_state["last_prediction"] = result
                st.session_state["last_inputs"] = values

                st.success("Prediction completed successfully.")

                st.subheader("Prediction Results")

                result_row = result.iloc[0]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Heat Transfer Rate",
                    f"{result_row['Q_W']:,.3f} W"
                )

                col2.metric(
                    "Hot Outlet Temperature",
                    f"{result_row['Th_out_C']:,.3f} °C"
                )

                col3.metric(
                    "Cold Outlet Temperature",
                    f"{result_row['Tc_out_C']:,.3f} °C"
                )

                st.divider()

                st.subheader("Complete Model Predictions")

                results_display = pd.DataFrame({
                    "Predicted Parameter": [
                        TARGET_LABELS.get(x, x)
                        for x in TARGETS
                    ],
                    "Model Output": [
                        result_row[x]
                        for x in TARGETS
                    ],
                    "Unit": [
                        {
                            "Q_W": "W",
                            "Th_out_C": "°C",
                            "Tc_out_C": "°C",
                            "dp_h_Pa": "Pa",
                            "dp_c_Pa": "Pa",
                            "Nu_h": "-",
                            "Nu_c": "-",
                            "h_h_W_m2K": "W/m²K",
                            "h_c_W_m2K": "W/m²K"
                        }.get(x, "")
                        for x in TARGETS
                    ]
                })

                st.dataframe(
                    results_display,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Prediction Results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="openthermoai_prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.caption(
                    "Predictions are model estimates, not "
                    "independent experimental measurements."
                )

            except Exception as error:

                st.error("Prediction failed.")
                st.exception(error)


# =====================================================
# 10. SCENARIO ANALYSIS
# =====================================================

elif page == "Scenario Analysis":

    st.header("Batch Scenario Analysis")

    st.write(
        "Evaluate multiple heat exchanger operating scenarios "
        "using the existing trained model."
    )

    st.info(
        "Upload a CSV containing all 10 model input columns. "
        "The application will predict all 9 outputs for each row."
    )

    st.subheader("Required CSV Input Columns")

    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Upload operating scenarios CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            scenario_data = pd.read_csv(uploaded_file)

            missing_features = [
                feature for feature in FEATURES
                if feature not in scenario_data.columns
            ]

            if missing_features:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_features)
                )

            elif scenario_data.empty:

                st.warning("The uploaded CSV contains no data.")

            else:

                input_data = scenario_data[FEATURES].copy()

                input_data = input_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if input_data.isna().any().any():

                    st.error(
                        "The input CSV contains missing or "
                        "non-numeric feature values."
                    )

                elif not np.isfinite(
                    input_data.to_numpy(dtype=float)
                ).all():

                    st.error(
                        "The CSV contains infinite or invalid values."
                    )

                elif (
                    input_data[POSITIVE_FEATURES] <= 0
                ).any().any():

                    st.error(
                        "Mass flow, area, U, diameters and lengths "
                        "must be positive."
                    )

                else:

                    st.subheader("Uploaded Scenario Data")

                    st.dataframe(
                        scenario_data.head(20),
                        use_container_width=True
                    )

                    if st.button(
                        "Run Scenario Predictions",
                        type="primary"
                    ):

                        predictions = make_predictions(
                            input_data
                        )

                        combined = pd.concat(
                            [
                                scenario_data.reset_index(drop=True),
                                predictions.reset_index(drop=True)
                            ],
                            axis=1
                        )

                        st.success(
                            f"Successfully evaluated "
                            f"{len(combined)} scenarios."
                        )

                        st.subheader("Scenario Prediction Results")

                        st.dataframe(
                            combined,
                            use_container_width=True
                        )

                        st.download_button(
                            "Download All Scenario Results",
                            data=combined.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="openthermoai_scenario_results.csv",
                            mime="text/csv"
                        )

        except Exception as error:

            st.error("Could not process the uploaded CSV.")
            st.exception(error)

    else:

        if SCENARIOS_PATH.exists():

            st.subheader("Existing Scenario Dataset")

            try:
                existing_scenarios = pd.read_csv(
                    SCENARIOS_PATH
                )

                st.dataframe(
                    existing_scenarios.head(20),
                    use_container_width=True
                )

                st.download_button(
                    "Download Existing Scenario Dataset",
                    data=existing_scenarios.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="openthermoai_scenarios.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.warning(
                    f"Could not read scenario dataset: {error}"
                )

        else:

            st.warning(
                "No existing scenario dataset was found. "
                "You can upload a CSV to begin scenario analysis."
            )

# =====================================================
# 12. ABOUT OPEN THERMO AI
# =====================================================

elif page == "About OpenThermoAI":

    st.header("About OpenThermoAI")

    st.markdown(
        """
        **OpenThermoAI** is a machine-learning application
        for heat exchanger thermal and hydraulic performance
        prediction.

        The application uses an existing trained
        Extra Trees regression model.
        """
    )

    st.subheader("Model Information")

    information = {
        "Model architecture": type(model).__name__,
        "Input features": len(FEATURES),
        "Prediction targets": len(TARGETS),
        "Training data type": bundle.get(
            "training_data_type",
            "Not specified"
        ),
        "Selected model": bundle.get(
            "selected_model",
            "Not specified"
        ),
        "Random seed": bundle.get(
            "random_seed",
            "Not specified"
        ),
        "Model file size": f"{model_size:.2f} MB"
    }

    st.table(
        pd.DataFrame(
            list(information.items()),
            columns=["Property", "Value"]
        )
    )

    st.subheader("Input Features")

    st.write(
        pd.DataFrame({
            "Feature": FEATURES,
            "Description": [
                FEATURE_LABELS.get(x, x)
                for x in FEATURES
            ]
        })
    )

    st.subheader("Prediction Targets")

    st.write(
        pd.DataFrame({
            "Target": TARGETS,
            "Description": [
                TARGET_LABELS.get(x, x)
                for x in TARGETS
            ]
        })
    )

    st.subheader("Model Assumptions")

    assumptions = bundle.get("assumptions")

    if assumptions is not None:
        st.write(assumptions)
    else:
        st.info("No model assumptions were stored in the bundle.")


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "OpenThermoAI | Machine Learning for Thermal System "
    "Performance Prediction"
)

st.caption(
    "Predictions should be interpreted within the model's "
    "training domain and independently validated before "
    "engineering or operational decisions."
)
