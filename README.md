# 🌡️ OpenThermoAI
### Machine Learning-Based Heat Exchanger Performance Prediction and Analysis

**Artificial Intelligence | Computational Thermofluids | Heat Transfer | Machine Learning | Energy Systems**

OpenThermoAI is a machine learning-powered engineering application developed to predict the thermal and hydraulic performance of heat exchanger systems.

The platform integrates an **Extra Trees Regression model** with an interactive Streamlit dashboard to estimate important heat exchanger performance parameters, including heat-transfer rate, outlet temperatures, pressure drops, Nusselt numbers, and convective heat-transfer coefficients.

The project demonstrates the integration of mechanical engineering principles, thermal-fluid sciences, and artificial intelligence for data-driven thermal system analysis and optimization.

---

## 🚀 1. Project Overview

Heat exchangers are essential components in power plants, industrial processes, refrigeration systems, chemical processing facilities, and energy conversion systems.

Their performance depends on several operating and geometric parameters, including:

- Hot- and cold-fluid inlet temperatures.
- Fluid mass flow rates.
- Heat-transfer surface area.
- Overall heat-transfer coefficient.
- Hydraulic diameters.
- Heat exchanger lengths.

Traditional heat exchanger analysis often requires iterative calculations, empirical correlations, and computational fluid dynamics simulations.

OpenThermoAI explores how machine learning can complement these engineering methods by developing a predictive framework that estimates multiple thermal and hydraulic performance indicators from specified operating conditions.

The application provides an interactive environment for individual predictions and batch evaluation of different operating scenarios.

## 🎯 2. Project Objectives

1. Develop a machine learning model for heat exchanger performance prediction.
2. Predict multiple thermal and hydraulic performance parameters simultaneously.
3. Investigate the relationship between operating conditions and heat exchanger performance.
4. Provide an interactive engineering prediction dashboard.
5. Evaluate multiple heat exchanger operating scenarios.
6. Establish a foundation for machine learning-assisted thermal system optimization.
7. Explore the integration of machine learning with conventional thermofluid engineering and future CFD validation.

## ✨ 3. Key Features

### A. Interactive Engineering Dashboard

The Streamlit application provides four main navigation sections:

- Home
- Performance Prediction
- Scenario Analysis
- About OpenThermoAI

### B. Heat Exchanger Performance Prediction

Users can enter operating and geometric parameters to obtain predictions for nine engineering performance indicators.

### C. Batch Scenario Analysis

- Upload heat exchanger operating scenarios through CSV files.
- Validate required input columns.
- Evaluate multiple operating conditions.
- Generate predictions for all nine target variables.
- Export combined input and prediction results as CSV.

### D. Model Information and Inspection

- Displays the selected model architecture.
- Identifies the number of input features.
- Identifies the number of prediction targets.
- Displays model configuration information.
- Reports training-data information when available in the model bundle.

### E. Prediction Result Export

Prediction results can be downloaded for further engineering analysis, visualization, and reporting.

---

## 🧮 4. Machine Learning Model

<text>**Selected algorithm: Extra Trees Regressor (Extremely Randomized Trees)**</text>

The application uses a previously trained Extra Trees regression model to estimate multiple heat exchanger performance parameters.

Extra Trees is an ensemble learning algorithm that combines multiple randomized decision trees to produce predictions.

The model is suitable for exploring nonlinear relationships between heat exchanger operating conditions and thermal-fluid performance indicators.

### Model configuration

| Parameter | Configuration |
|---|---|
| Model architecture | Extra Trees Regressor |
| Learning approach | Supervised learning |
| Problem type | Multi-output regression |
| Input features | 10 |
| Prediction targets | 9 |
| Model serialization | Joblib |
| Application framework | Streamlit |

### Input features

The model uses ten engineering input parameters.

| Variable | Description | Unit |
|---|---|---|
| Th_in_C | Hot-fluid inlet temperature | °C |
| Tc_in_C | Cold-fluid inlet temperature | °C |
| mdot_h_kg_s | Hot-fluid mass flow rate | kg/s |
| mdot_c_kg_s | Cold-fluid mass flow rate | kg/s |
| area_m2 | Heat-transfer area | m² |
| U_W_m2K | Overall heat-transfer coefficient | W/m²K |
| Dh_h_m | Hot-side hydraulic diameter | m |
| Dh_c_m | Cold-side hydraulic diameter | m |
| L_h_m | Hot-side length | m |
| L_c_m | Cold-side length | m |

### Prediction targets

The model predicts nine thermal and hydraulic performance parameters.

| Variable | Description | Unit |
|---|---|---|
| Q_W | Heat-transfer rate | W |
| Th_out_C | Hot-fluid outlet temperature | °C |
| Tc_out_C | Cold-fluid outlet temperature | °C |
| dp_h_Pa | Hot-side pressure drop | Pa |
| dp_c_Pa | Cold-side pressure drop | Pa |
| Nu_h | Hot-side Nusselt number | Dimensionless |
| Nu_c | Cold-side Nusselt number | Dimensionless |
| h_h_W_m2K | Hot-side heat-transfer coefficient | W/m²K |
| h_c_W_m2K | Cold-side heat-transfer coefficient | W/m²K |

---

## 🏗️ 5. System Architecture

```text
       HEAT EXCHANGER INPUTS
                  |
                  v
       THERMAL CONDITIONS
       - Hot inlet temperature
       - Cold inlet temperature
       - Hot mass flow rate
       - Cold mass flow rate
                  |
                  v
       GEOMETRIC PARAMETERS
       - Heat transfer area
       - Hydraulic diameters
       - Heat exchanger lengths
       - Overall heat transfer coefficient
                  |
                  v
          INPUT VALIDATION
                  |
                  v
       FEATURE ORDERING AND
         DATA PREPARATION
                  |
                  v
       PRETRAINED EXTRA TREES
          REGRESSION MODEL
                  |
                  v
        MULTI-OUTPUT PREDICTION
                  |
          +-------+-------+
          |               |
          v               v
   INDIVIDUAL           BATCH
   PREDICTION          ANALYSIS
          |               |
          +-------+-------+
                  |
                  v
        NINE PERFORMANCE OUTPUTS
                  |
                  v
        STREAMLIT DASHBOARD
                  |
                  v
       RESULTS VISUALIZATION
          AND CSV EXPORT
```

## 🔬 6. Engineering Applications

OpenThermoAI provides a foundation for machine learning applications in:

- Industrial heat exchanger design.
- Thermal system performance assessment.
- Power plant heat recovery systems.
- Cooling and heating applications.
- Industrial energy efficiency.
- Thermal equipment performance monitoring.
- Heat exchanger operating-condition analysis.
- Data-driven thermal system optimization.

## 📊 7. Scenario Analysis

The application supports batch evaluation of heat exchanger operating conditions.

Users can upload a CSV file containing the ten required model input columns.

Example CSV structure:

```csv
Th_in_C,Tc_in_C,mdot_h_kg_s,mdot_c_kg_s,area_m2,U_W_m2K,Dh_h_m,Dh_c_m,L_h_m,L_c_m
80,25,1.5,1.2,10,500,0.02,0.02,5,5
90,30,1.8,1.4,12,550,0.02,0.02,6,6
```

These values are illustrative input examples, not validated operating recommendations.

The application processes the uploaded scenarios and generates nine predicted performance outputs for every valid row.

Results can be downloaded for additional engineering analysis.

## 🛠️ 8. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming |
| Streamlit | Interactive web application |
| Scikit-learn | Extra Trees regression |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Joblib | Model serialization and loading |
| gdown | Google Drive model retrieval |
| GitHub | Version control |
| Streamlit Community Cloud | Application deployment |

## 📂 9. Repository Structure

```text
openthermo-app/
│
├── .devcontainer/
│
├── streamlit_app.py
│
├── openthermoai_model.joblib
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── LICENSE
```

### File descriptions

- `streamlit_app.py`: Main application and prediction interface.
- `openthermoai_model.joblib`: Trained machine learning model bundle.
- `requirements.txt`: Project dependencies.
- `.devcontainer/`: Development environment configuration.
- `.gitignore`: Files excluded from version control.
- `LICENSE`: MIT license.

The application can retrieve the trained model from Google Drive when the local model file is unavailable.

## ⚙️ 10. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/jegedeolakunle21-art/openthermo-app.git
```

### Step 2: Navigate into the project directory

```bash
cd openthermo-app
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

## ☁️ 11. Deployment

The application is designed for deployment using Streamlit Community Cloud.

Deployment steps:

1. Push the repository to GitHub.
2. Visit https://share.streamlit.io/
3. Sign in using your GitHub account.
4. Select the `openthermo-app` repository.
5. Select the `main` branch.
6. Set the application entry point to `streamlit_app.py`.
7. Click Deploy.

Ensure that the trained model file or its configured Google Drive download source is accessible.

## 📈 12. Future Research and Development

Future improvements will focus on expanding the predictive capabilities and engineering validation of OpenThermoAI.

### Machine Learning Improvements

- Compare Extra Trees, Random Forest, XGBoost, and Artificial Neural Networks.
- Perform hyperparameter optimization.
- Introduce cross-validation.
- Evaluate model performance using MAE, RMSE, and R².
- Implement SHAP-based explainability.
- Investigate model uncertainty and generalization.

### CFD Integration

- Generate independent CFD simulation datasets using ANSYS Fluent or CFX.
- Investigate turbulent flow and heat-transfer characteristics.
- Compare machine learning predictions with CFD results.
- Evaluate model accuracy across different flow regimes.
- Develop CFD-informed surrogate models.

### Thermal System Optimization

- Optimize heat-transfer rate.
- Minimize pressure losses.
- Investigate the effects of mass flow rate and heat exchanger geometry.
- Develop multi-objective optimization algorithms.
- Introduce engineering constraints into the optimization framework.

### Advanced Dashboard Development

- Interactive performance comparison.
- Sensitivity analysis.
- Optimization visualization.
- Model validation dashboard.
- Downloadable engineering reports.
- Integration of real industrial operating data.

## ⚠️ 13. Model Limitations and Engineering Validation

OpenThermoAI produces machine learning estimates based on its training data.

Important considerations:

- Predictions should remain within the model's established training domain.
- The model's training-data provenance and operating ranges should be examined before practical application.
- Predictions are not independent experimental measurements.
- CFD or experimental validation is required before using the model for engineering design or operational decisions.
- The model does not replace established heat-transfer calculations, engineering standards, or professional engineering judgment.

## 👨‍💻 14. Author

**Olakunle Jegede**

Mechanical Engineer | Thermofluids | Computational Fluid Dynamics | Artificial Intelligence | Machine Learning | Energy Systems

### Research Interests

- Thermodynamics
- Heat Transfer and Fluid Mechanics
- Computational Fluid Dynamics
- Machine Learning-Enhanced CFD
- Thermal System Optimization
- Energy Systems Engineering
- Industrial AI and Predictive Analytics

## 📜 15. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ⭐ Acknowledgements

OpenThermoAI represents an ongoing effort to integrate mechanical engineering, computational thermofluids, and machine learning to develop intelligent predictive tools for thermal engineering applications.

If you find this project interesting, please consider giving the repository a star.

---

**Project:** OpenThermoAI

**Repository:** https://github.com/jegedeolakunle21-art/openthermo-app

**Application:** Machine Learning-Based Heat Exchanger Performance Prediction and Analysis

**Primary Algorithm:** Extra Trees Regressor

**Engineering Domain:** Heat Transfer and Fluid Mechanics
