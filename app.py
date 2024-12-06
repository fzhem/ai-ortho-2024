import base64

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from extended_pipeline import ExtendedPipeline
from hierarchical_model import HierarchicalModel

binary_pipeline: ExtendedPipeline = joblib.load("models/normal_v_abnormal.pkl")
X_binary_rfe_train = pd.read_parquet("parquets/X_binary_rfe_train.parquet")
X_multi_class_train = pd.read_parquet("parquets/X_multi_class_train.parquet")

st.set_page_config(page_title="AI in orthopaedics 2024", page_icon="🦴", layout="wide")

col1, col2 = st.columns([1, 1], vertical_alignment="center")
prediction = None
predictions = None
class_normal_prediction = None
class_abnormal_prediction = None
fig = None
with col1:
    st.title("Spine Parameters Prediction with Multiple Models")

    model_choice = st.radio("Select a Model", ["Binary", "Multiclass"])

    pelvic_incidence = st.number_input("Pelvic Incidence", value=34.76)
    pelvic_tilt = st.number_input("Pelvic Tilt", value=2.63)
    lumbar_lordosis_angle = st.number_input("Lumbar Lordosis Angle", value=29.50)
    sacral_slope = st.number_input("Sacral Slope", value=32.12)
    pelvic_radius = st.number_input("Pelvic Radius", value=127.14)
    degree_spondylolisthesis = st.number_input("Degree Spondylolisthesis", value=-0.46)

    if st.button("Predict"):
        match model_choice:
            case "Binary":
                scaler = binary_pipeline.named_steps["scaler"]
                input_data = pd.DataFrame(
                    [
                        [
                            pelvic_tilt,
                            sacral_slope,
                            pelvic_radius,
                            degree_spondylolisthesis,
                        ]
                    ],
                    columns=[
                        "pelvic_tilt",
                        "sacral_slope",
                        "pelvic_radius",
                        "degree_spondylolisthesis",
                    ],
                )
                prediction = binary_pipeline.predict(
                    pd.DataFrame(
                        input_data,
                        columns=[
                            "pelvic_tilt",
                            "sacral_slope",
                            "pelvic_radius",
                            "degree_spondylolisthesis",
                        ],
                    )
                )[0]

                y_pred = binary_pipeline.predict_proba(
                    pd.DataFrame(
                        input_data,
                        columns=[
                            "pelvic_tilt",
                            "sacral_slope",
                            "pelvic_radius",
                            "degree_spondylolisthesis",
                        ],
                    )
                )
                class_abnormal_prediction = y_pred[:, 0].item()
                class_normal_prediction = y_pred[:, 1].item()

                explainer = shap.LinearExplainer(
                    ExtendedPipeline(binary_pipeline),
                    scaler.transform(X_binary_rfe_train),
                    feature_names=X_binary_rfe_train.columns,
                )
                shap_values = explainer(scaler.transform(input_data))
            case _:
                hierarchical_pipeline = HierarchicalModel.load(
                    "models/hierarchical.pkl"
                )
                COLUMNS = [
                    "pelvic_incidence",
                    "pelvic_tilt",
                    "lumbar_lordosis_angle",
                    "sacral_slope",
                    "pelvic_radius",
                    "degree_spondylolisthesis",
                ]
                input_data = pd.DataFrame(
                    [
                        [
                            pelvic_incidence,
                            pelvic_tilt,
                            lumbar_lordosis_angle,
                            sacral_slope,
                            pelvic_radius,
                            degree_spondylolisthesis,
                        ]
                    ],
                    columns=COLUMNS,
                )
                predictions = hierarchical_pipeline.predict(input_data)
                prediction = predictions["final_prediction"].iloc[0]

                hernia_features = X_multi_class_train[
                    [
                        "pelvic_tilt",
                        "lumbar_lordosis_angle",
                        "sacral_slope",
                        "pelvic_radius",
                    ]
                ]

                if prediction != 'Spondylolisthesis':

                    hernia_explainer = shap.LinearExplainer(
                        ExtendedPipeline(hierarchical_pipeline.hernia_pipeline),
                        hierarchical_pipeline.scale_hernia(hernia_features),
                        feature_names=hernia_features.columns,
                    )
                    hernia_shap_values = hernia_explainer(hernia_features)

with col2:
    if prediction is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; margin-top: 20px;">
                Prediction from {model_choice}: {prediction}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if class_normal_prediction is not None and class_abnormal_prediction is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 18px; margin-top: 10px;">
                Class Normal prediction: {class_normal_prediction:.4f}<br>
                Class Abnormal prediction: {class_abnormal_prediction:.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if predictions is not None:
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 18px; margin-top: 10px;">
                Class Normal prediction: {predictions['normal_proba'].iloc[0]:.4f}<br>
                Class Hernia prediction: {predictions['hernia_proba'].iloc[0]:.4f}<br>
                Class Spondylolisthesis prediction: {predictions['spondy_proba'].iloc[0]:.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )
    if "shap_values" in locals() and shap_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots._waterfall.waterfall_legacy(
            shap_values.base_values[0],
            shap_values.values[0],
            feature_names=[
                f"pelvic_tilt: {pelvic_tilt}",
                f"sacral_slope: {sacral_slope}",
                f"pelvic_radius: {pelvic_radius}",
                f"degree_spondylolisthesis: {degree_spondylolisthesis}",
            ],
        )
        st.pyplot(fig)

    if "hernia_shap_values" in locals() and hernia_shap_values:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots._waterfall.waterfall_legacy(
            hernia_shap_values.base_values[0],
            hernia_shap_values.values[0],
            feature_names=[
                f"pelvic_tilt: {pelvic_tilt}",
                f"lumbar_lordosis_angle: {lumbar_lordosis_angle}",
                f"sacral_slope: {sacral_slope}",
                f"pelvic_radius: {pelvic_radius}",
            ],
            show=False,
        )
        st.pyplot(fig)