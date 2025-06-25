import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="AIS Prognosis App", layout="centered")
st.title("🧠 AIS 90-Day Outcome Prediction (mRS 3–6)")

# 加载模型和标准化器
model = joblib.load('model_06_25.pkl')
scaler = joblib.load('scaler_06_25.pkl')

feature_names = ['Age', 'Baseline NIHSS', 'Baseline mRS', 'Glucose', 'WBC', 'hs-CRP']

# 用户输入
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 18, 100, 62)
    Baseline_mRS = st.number_input("Baseline mRS", 0, 5, 3)
    WBC = st.number_input("WBC (×10⁹/L)", 2.0, 30.0, 7.0)
with col2:
    Baseline_NIHSS = st.number_input("Baseline NIHSS", 0, 42, 6)
    Glucose = st.number_input("Glucose (mmol/L)", 2.2, 32.0, 6.5)
    hs_CRP = st.number_input("hs-CRP (mg/L)", 0.10, 200.00, 5.25)

# 构造输入
X_input_df = pd.DataFrame([[Age, Baseline_NIHSS, Baseline_mRS, Glucose, WBC, hs_CRP]], columns=feature_names)
X_scaled = scaler.transform(X_input_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# 预测与解释
if st.button("Predict"):
    proba = model.predict_proba(X_scaled_df)[0]

    st.markdown("### Prediction Results")
    st.markdown(f"- **Predicted Probability of mRS 3–6**: {proba[1]*100:.2f}%")
    st.markdown(f"- **Probability of Good Outcome**: {proba[0]*100:.2f}%")

    with st.spinner("Generating SHAP force plot..."):
        try:
            explainer = joblib.load("shap_explainer_06_25.pkl")
        except Exception as e:
            st.warning(f"SHAP explainer load failed: {e}")
            masker = shap.maskers.Independent(X_scaled_df)
            explainer = shap.LinearExplainer(model, masker=masker)

        shap_values = explainer(X_scaled_df)

        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        shap_contribs = shap_values.values[0]

        # 画 force plot
        plt.clf()
        fig = plt.figure(figsize=(12, 3), dpi=600)
        shap.force_plot(
            base_value=base_value,
            shap_values=shap_contribs,
            features=X_scaled_df.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        logit_val = base_value + shap_contribs.sum()
        st.caption(f"base: {base_value:.3f} + sum(SHAP): {shap_contribs.sum():.3f} = f(x): {logit_val:.3f}")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=600)
        plt.close()
        st.image(buf.getvalue(), caption="SHAP Force Plot", use_container_width=True)
