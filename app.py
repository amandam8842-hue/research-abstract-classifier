```python
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Research Abstract Classifier",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Article Classification")

st.write(
    "Predict whether a research abstract belongs to "
    "Science, Business, or Social Science."
)

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")


model = load_model()

# Model information
with st.expander("Model information"):
    st.write("**Best model:** SVM (Word + Character TF-IDF)")
    st.write("**Test accuracy:** 92%")
    st.write("**Macro F1-score:** 0.92")
    st.write("**Training dataset:** 8,983 abstracts")


# Default example
st.subheader("Default example")

default_abstract = (
    "This study investigates how changes in monetary policy "
    "influence investment decisions and firm performance "
    "across different markets."
)

st.write(default_abstract)

st.success(
    f"Predicted label: {model.predict([default_abstract])[0]}"
)


# Single abstract prediction
st.subheader("Try your own abstract")

abstract = st.text_area(
    "Paste a research abstract",
    height=220
)

if st.button("Predict class"):
    if len(abstract.strip()) < 30:
        st.warning(
            "Please enter a longer abstract "
            "(at least 30 characters)."
        )
    else:
        prediction = model.predict([abstract])[0]
        st.success(f"Predicted label: **{prediction}**")


st.divider()


# Batch prediction
st.subheader("Batch prediction")

uploaded = st.file_uploader(
    "Upload a CSV with an 'abstract' column",
    type=["csv"]
)

if uploaded is not None:

    try:
        data = pd.read_csv(uploaded)

        if "abstract" not in data.columns:
            st.error(
                "Your CSV must contain a column named 'abstract'."
            )

        else:
            valid = (
                data["abstract"].fillna("").astype(str).str.strip().str.len()
                >= 30
            )

            data["predicted_label"] = "Invalid or empty abstract"

            if valid.any():
                data.loc[valid, "predicted_label"] = model.predict(
                    data.loc[valid, "abstract"].astype(str)
                )

            st.success(
                f"Processed {len(data)} records."
            )

            st.dataframe(data.head(100))

            st.download_button(
                "Download predictions",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Unable to process CSV: {e}")


st.divider()

st.caption(
    "Educational research tool. Predictions reflect "
    "the quality and biases of the training labels."
)
```
