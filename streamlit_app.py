import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64

# 设置页面
st.set_page_config(page_title="AIS Prognosis App", layout="centered")
st.title("🧠 AIS 90-day Prognosis Predictor (mRS 3–6)")

# 加载模型与标准化器
model = joblib.load('model_6features_calibrated.pkl')
scaler = joblib.load('scaler_6features.pkl')

# 定义特征名称（与训练模型保持一致）
feature_names = ['Age', 'Baseline NIHSS', 'Baseline mRS', 'Glucose', 'WBC', 'hs-CRP']

# 交互式输入表单
with st.form("prediction_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=62)
    NIHSS = st.number_input("Baseline NIHSS", min_value=0, max_value=42, value=6)
    mRS = st.number_input("Baseline mRS", min_value=0, max_value=5, value=3)
    Glucose = st.number_input("Glucose (mmol/L)", min_value=2.2, max_value=32.0, value=6.5)
    WBC = st.number_input("WBC (×10⁹/L)", min_value=2.00, max_value=30.00, value=7.00)
    hs_CRP = st.number_input("hs-CRP (mg/L)", min_value=0.00, max_value=200.00, value=5.25)

    submitted = st.form_submit_button("Predict")

# 预测逻辑
if submitted:
    feature_values = [Age, NIHSS, mRS, Glucose, WBC, hs_CRP]
    input_array = np.array([feature_values])
    input_scaled = scaler.transform(input_array)

    predicted_proba = model.predict_proba(input_scaled)[0]
    risk = predicted_proba[1] * 100

    st.success(f"Predicted risk of mRS 3–6 at 90 days: **{risk:.1f}%**")

    # 解释模型（使用 SHAP linear explainer）
    explainer = shap.Explainer(model, masker=scaler.transform, algorithm="linear")
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    # 保存 force 图为图像（由于 Streamlit 不支持 JS force plot）
    plt.figure()
    shap.plots.force(shap_values[0], matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", dpi=300)
    st.image("shap_force_plot.png", caption="SHAP Force Plot", use_column_width=True)
