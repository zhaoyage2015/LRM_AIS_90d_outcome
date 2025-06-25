import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# ----------------------------
# 页面设置
# ----------------------------
st.set_page_config(page_title="AIS Prognosis App", layout="centered")
st.title("🧠 AIS 90-Day Outcome Prediction (mRS 3–6)")

# ----------------------------
# 加载模型与标准化器
# ----------------------------
model = joblib.load('model_06_25.pkl')
scaler = joblib.load('scaler_06_25.pkl')

# 若存在训练时保存的特征顺序，则加载
try:
    feature_names = joblib.load("feature_names_used.pkl")
except:
    feature_names = ['Age', 'Baseline NIHSS', 'Baseline mRS', 'Glucose', 'WBC', 'hs-CRP']

# ----------------------------
# 构建输入框（与特征顺序一致）
# ----------------------------
input_values = []
input_fields = {
    'Age': (18, 100, 62),
    'Baseline NIHSS': (0, 42, 6),
    'Baseline mRS': (0, 5, 3),
    'Glucose': (2.2, 32.0, 6.5),
    'WBC': (2.0, 30.0, 7.0),
    'hs-CRP': (0.1, 200.0, 5.25)
}

col1, col2 = st.columns(2)
for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        min_val, max_val, default = input_fields[feature]
        val = st.number_input(feature, min_value=min_val, max_value=max_val, value=default, key=feature)
        input_values.append(val)

# ----------------------------
# 标准化输入
# ----------------------------
X_input_df = pd.DataFrame([input_values], columns=feature_names)
X_scaled = scaler.transform(X_input_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# ----------------------------
# 预测 + SHAP解释
# ----------------------------
if st.button("Predict"):
    proba = model.predict_proba(X_scaled_df)[0]

    st.markdown("### Prediction Results")
    st.markdown(f"- **Predicted Probability of mRS 3–6**: {proba[1]*100:.2f}%")
    st.markdown(f"- **Probability of Good Outcome**: {proba[0]*100:.2f}%")

    with st.spinner("Generating SHAP force plot..."):
        try:
            explainer = joblib.load("shap_explainer_06_25.pkl")
        except Exception as e:
            st.warning(f"SHAP explainer not found or failed to load. Using fallback explainer. Error: {e}")
            from shap.maskers import Independent
            masker = Independent(X_scaled_df)
            explainer = shap.LinearExplainer(model, masker=masker)

        shap_values = explainer(X_scaled_df)

        # 处理 SHAP 值结构
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        shap_contributions = shap_values.values[0][:, 1] if shap_values.values.ndim == 3 else shap_values.values[0]

        # 可视化
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

        logit_val = base_value + shap_contributions.sum()
        st.caption(f"base: {base_value:.3f} + sum(SHAP): {shap_contributions.sum():.3f} = f(x): {logit_val:.3f}")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=600)
        plt.close()
        st.image(buf.getvalue(), caption="SHAP Force Plot", use_container_width=True)


    
