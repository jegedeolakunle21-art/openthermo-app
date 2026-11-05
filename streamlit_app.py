
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE = Path(__file__).resolve().parent

MODEL_DIR = BASE / "models"
MODEL_PATH = MODEL_DIR / "openthermoai_model.joblib"
TEMP_MODEL_PATH = MODEL_DIR / "openthermoai_model.download"

METRICS_PATH = BASE / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE / "data" / "optimization_scenarios.csv"

GOOGLE_DRIVE_FILE_ID = "18ZPMFhU303wK_-9Hz-gn4aSwCr-IGHRj"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="OpenThermoAI",
    page_icon="🌡️",
    layout="wide"
)

# ==========================================
# 3. LOAD EXISTING TRAINED MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_bundle():

    # Download only when the existing model is missing
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        print("Downloading existing OpenThermoAI model...")

        # Remove any incomplete previous download
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()

        downloaded_file = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(TEMP_MODEL_PATH),
            quiet=False
        )

        if (
            downloaded_file is None
            or not TEMP_MODEL_PATH.exists()
            or TEMP_MODEL_PATH.stat().st_size == 0
        ):
            raise FileNotFoundError(
                "Google Drive model download failed. "
                "Check the sharing permissions."
            )

        # Verify the downloaded file
        if TEMP_MODEL_PATH.stat().st_size < 1_000_000:
            raise ValueError(
                "Downloaded model file is unexpectedly small."
            )

        # Move the completed download to its final location
        TEMP_MODEL_PATH.replace(MODEL_PATH)

    # Confirm model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    print("Loading existing trained model...")

    # Load the original trained model
    bundle = joblib.load(MODEL_PATH)

    # Validate the model bundle
    if not isinstance(bundle, dict):
        raise TypeError(
            "The saved model bundle is not a dictionary."
        )

    required_keys = ["model", "features", "targets"]

    missing_keys = [
        key for key in required_keys if key not in bundle
    ]

    if missing_keys:
        raise KeyError(
            f"Missing model bundle components: {missing_keys}"
        )

    print("OpenThermoAI model loaded successfully.")

    return bundle


# ==========================================
# 4. LOAD MODEL WITH ERROR HANDLING
# ==========================================

try:
    with st.spinner(
        "Loading your existing OpenThermoAI trained model..."
    ):
        bundle = load_bundle()

except Exception as e:
    st.error(
        "OpenThermoAI could not load its existing trained model."
    )

    st.error(f"Error type: {type(e).__name__}")
    st.exception(e)

    st.stop()


# ==========================================
# 5. EXTRACT MODEL COMPONENTS
# ==========================================

model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]

# ==========================================
# MODEL CONFIGURATION INSPECTION
# ==========================================

st.title("OpenThermoAI Configuration Inspector")

st.subheader("1. Model information")

st.write("Model type:", type(model).__name__)

st.write("Model bundle keys:", list(bundle.keys()))

st.subheader("2. Input features")

st.write("Features:", FEATURES)

st.write("Number of features:", len(FEATURES))

st.subheader("3. Prediction targets")

st.write("Targets:", TARGETS)

st.write("Number of targets:", len(TARGETS))

st.subheader("4. Fitted model configuration")

st.write(
    "Expected input features:",
    getattr(model, "n_features_in_", "Not available")
)

st.write(
    "Feature names:",
    getattr(model, "feature_names_in_", "Not available")
)

st.write(
    "Number of outputs:",
    getattr(model, "n_outputs_", "Not available")
)

st.success("Model configuration inspection completed.")

st.success("OpenThermoAI model is ready!")

st.caption(
    f"Loaded existing model: "
    f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
)

# ==========================================
# 6. CONTINUE WITH YOUR EXISTING APPLICATION
# ==========================================

# Keep the rest of your original application below this line.
# Preserve your prediction, optimization, validation,
# dashboard, charts and CSV download sections.
