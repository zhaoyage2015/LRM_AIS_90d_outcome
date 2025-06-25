import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 页面设置
st.set_page_config(page_title="AIS Prognosis App", layout="centered")
st.title("🧠 AIS 90-Day Outcome Prediction (mRS 3–6)")

# 加载模型与标准化器
model = joblib.load('model_06_25.pkl')
scaler = joblib.load('scaler_06_25.pkl')

# 特征名
feature_names = ['Age', 'Baseline NIHSS', 'Baseline mRS', 'Glucose', 'WBC', 'hs-CRP']

# 三行两列的输入模块布局
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=62)
    Baseline_mRS = st.number_input("Baseline mRS", min_value=0, max_value=5, value=3)
    WBC = st.number_input("WBC (×10⁹/L)", min_value=2.0, max_value=30.0, value=7.0)
with col2:
    Baseline_NIHSS = st.number_input("Baseline NIHSS", min_value=0, max_value=42, value=6)
    Glucose = st.number_input("Glucose (mmol/L)", min_value=2.2, max_value=32.0, value=6.5)
    hs_CRP = st.number_input("hs-CRP (mg/L)", min_value=0.10, max_value=200.00, value=5.25)

# 构造输入
feature_values = [Age, Baseline_NIHSS, Baseline_mRS, Glucose, WBC, hs_CRP]
X_input = pd.DataFrame([feature_values], columns=feature_names)

# 预测与解释
X_scaled = scaler.transform(X_input)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# 预测与解释
if st.button("Predict"):
    proba = model.predict_proba(X_scaled_df)[0]

    st.markdown("### Prediction Results")
    st.markdown(f"- **Predicted Probability of mRS 3–6**: {proba[1]*100:.2f}%")
    st.markdown(f"- **Probability of Good Outcome**: {proba[0]*100:.2f}%")

    # SHAP Force Plot
    with st.spinner("Generating SHAP force plot..."):
        try:
            explainer = joblib.load("shap_explainer_06_25.pkl")
        except Exception as e:
            st.warning(f"SHAP explainer not found or failed to load. Using fallback explainer. Error: {e}")
            from shap.maskers import Independent
            masker = Independent(X_scaled_df)
            explainer = shap.LinearExplainer(model, masker=masker)

        shap_values = explainer(X_scaled_df)

        # Force plot - class 1 (mRS 3–6)
        base_value = explainer.expected_value[1]
        shap_contributions = shap_values.values[0][:, 1] if shap_values.values.ndim == 3 else shap_values.values[0]

        plt.clf()
        fig = plt.figure(figsize=(12, 3), dpi=600)
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_contributions,
            features=X_scaled_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        # Debug: 显示 base + sum(SHAP) = logit
        logit_val = base_value + shap_contributions.sum()
        st.caption(f"base: {base_value:.3f} + sum(SHAP): {shap_contributions.sum():.3f} = f(x): {logit_val:.3f}")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=600)
        plt.close()
        st.image(buf.getvalue(), caption="SHAP Force Plot", use_container_width=True)

