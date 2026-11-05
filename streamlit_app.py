
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

METRICS_PATH = BASE / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE / "data" / "optimization_scenarios.csv"

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
# 3. GOOGLE DRIVE MODEL DOWNLOAD
# ==========================================

GOOGLE_DRIVE_FILE_ID = "18ZPMFhU303wK_-9Hz-gn4aSwCr-IGHRj"

@st.cache_resource
def load_bundle():
@st.cache_resource
def load_bundle():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    st.info("Step 1: Checking model file...")

    if not MODEL_PATH.exists():
        st.info("Step 2: Downloading existing trained model...")

        downloaded = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(TEMP_MODEL_PATH),
            quiet=False
        )

        if not downloaded or not TEMP_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Google Drive model download failed."
            )

        TEMP_MODEL_PATH.replace(MODEL_PATH)

    st.info("Step 3: Checking downloaded model size...")

    model_size = MODEL_PATH.stat().st_size

    if model_size < 1_000_000:
        raise ValueError("Downloaded model file is unexpectedly small.")

    st.write(f"Model file size: {model_size / (1024 * 1024):.2f} MB")

    st.info("Step 4: Loading trained model into memory...")

    bundle = joblib.load(MODEL_PATH)

    st.info("Step 5: Model loaded successfully!")

    return bundle


bundle = load_bundle()

st.info("Step 6: Checking model components...")

model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]

st.success("OpenThermoAI model is ready!")


    # Download the existing model only if it is missing
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        st.info("Downloading the existing OpenThermoAI model...")

        TEMP_MODEL_PATH = MODEL_DIR / "openthermoai_model.download"

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
                "Check the file-sharing permissions."
            )

        # Move completed download to the expected model location
        TEMP_MODEL_PATH.replace(MODEL_PATH)

    # Load the original trained model
    return joblib.load(MODEL_PATH)


# ==========================================
# 4. LOAD MODEL BUNDLE
# ==========================================

try:
    bundle = load_bundle()

except Exception as e:
    st.error(
        "OpenThermoAI could not load its trained model. "
        "Check Google Drive sharing permissions and deployment logs."
    )
    st.error(f"Error type: {type(e).__name__}")
    st.stop()


# ==========================================
# 5. EXTRACT MODEL COMPONENTS
# ==========================================

model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]
