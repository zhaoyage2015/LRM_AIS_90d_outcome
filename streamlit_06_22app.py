# ↓↓↓ 你的完整 Streamlit 脚本内容粘贴在这里 ↓↓↓
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
model = joblib.load('model_06_22.pkl')
scaler = joblib.load('scaler_06_22.pkl')

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
X_scaled = scaler.transform(X_input)

# 预测与解释
# 预测与解释
if st.button("Predict"):
    proba = model.predict_proba(X_scaled)[0]
    pred_class = model.predict(X_scaled)[0]

    st.markdown("### Prediction Results")
    st.markdown(f"- **Predicted Probability of mRS 3–6**: {proba[1]*100:.2f}%")
    st.markdown(f"- **Probability of Good Outcome**: {proba[0]*100:.2f}%")

    # SHAP Force Plot
    with st.spinner("Generating SHAP force plot..."):
        try:
            # 使用显式背景数据，避免部署错误
            explainer = joblib.load("shap_explainer_06_22.pkl")
            shap_values = explainer(X_scaled)

            plt.clf()
            fig = plt.figure(figsize=(12, 3), dpi=600)

            shap.force_plot(
                base_value=explainer.expected_value[1],
                shap_values=shap_values.values[0],
                features=X_input.iloc[0],
                feature_names=feature_names,
                feature_display_values=X_input.iloc[0].values,
                matplotlib=True,
                show=False
            )

            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=600)
            plt.close()
            st.image(buf.getvalue(), caption="SHAP Force Plot", use_column_width=True)

        except Exception as e:
            import traceback
            st.error(f"SHAP explanation failed: {str(e)}")
            st.text(traceback.format_exc())
