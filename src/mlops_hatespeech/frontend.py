import requests
import streamlit as st
# from google.cloud import run_v2


def classify_text(text, backend):
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    url = f"{backend}/predict"  # nur /predict anhängen
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error calling backend: {e}")
        return None


def main():
    st.title("Text Classification")

    backend = "https://bento-app-178847025464.europe-west1.run.app"
    if backend is None:
        st.error("Backend URL not set!")
        return

    user_input = st.text_area("Enter text to classify")

    if st.button("Predict") and user_input.strip():
        result = classify_text(user_input, backend=backend)
        if result is not None:
            pred_class = result.get("predicted_class", None)
            if pred_class == "non-hate":
                st.success("This is NOT hate speech.")
            elif pred_class == "hate":
                st.error("This is hate speech!")
            else:
                st.warning(f"Unknown prediction class: {pred_class}")
        else:
            st.error("Failed to get prediction from backend.")


if __name__ == "__main__":
    main()
