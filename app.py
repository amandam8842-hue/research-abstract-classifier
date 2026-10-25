
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Research Abstract Classifier", page_icon="📚")
st.title("Research Abstract Classification")
st.write("Predict whether an abstract is labelled Science, Business, or Social Science.")

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")


model = load_model()

# Default abstract prediction
default_abstract = (
    "This study investigates how changes in monetary policy influence "
    "investment decisions and firm performance across different markets."
)

st.subheader("Default example")
st.write(default_abstract)
st.success(f"Predicted label: {model.predict([default_abstract])[0]}")

# User enters their own abstract
abstract = st.text_area("Paste a research abstract", height=220)


if st.button("Predict class"):
    if len(abstract.strip()) < 30:
        st.warning("Please enter a longer abstract (at least 30 characters).")
    else:
        st.success(f"Predicted label: **{model.predict([abstract])[0]}**")

st.divider()
st.subheader("Batch prediction")
uploaded = st.file_uploader("Upload a CSV with an 'abstract' column", type=["csv"])

if uploaded:
    data = pd.read_csv(uploaded)

    if "abstract" not in data.columns:
        st.error("Your CSV must have a column named 'abstract'.")
    else:
        data["predicted_label"] = model.predict(
            data["abstract"].fillna("").astype(str)
        )
        st.dataframe(data.head(100))
        st.download_button(
            "Download predictions",
            data.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )

st.caption("Educational baseline. Predictions reflect the quality and biases of the training labels.")
