import streamlit as st
import requests

URL = "http://fastapi:8000/predict" 

st.set_page_config(
    page_title="Hate Speech Classifier",
)

st.title("SafetyGaurd Rail For Indic Languages")
st.subheader("This application detects hate speech in indic languages")
st.markdown("For now the application supports Hindi and Telugu languages. More languages will be added soon.")

text_inp = st.text_area("Enter text to classify", placeholder="Type something here...")


if text_inp is not None:
    submit = st.button("Classify", type="primary",width="stretch")

    if submit:
        with st.spinner("Classifying..."):
            if text_inp.strip() == "":
                st.warning("Please enter some text to classify.")
            else:
                response = requests.post(URL, json={"text": text_inp}, timeout=20)
                detected_lang = response.json().get("Detected Language")

                if response.json().get("Detected Language")  not in ['hi', 'te']:
                    st.error(f"Unsupported language detected: {detected_lang}. Only Hindi and Telugu are supported.")

                elif response.status_code == 200:
                    result = response.json()
                    label = result['predicted_label']
                    ans = None
                    print(label)
                    if label == 'NOT' or label == "non-hate":
                        ans = "The text is NOT hate speech."
                    else:
                        ans = "The text is hate speech."

                    st.success(f"Predicted Label: {ans}")
                    st.info(f"Confidence: {result['confidence']:.4f}")
                    st.info(f"Detected Language: {result['Detected Language']}")


                else:
                    st.error("Error in classification. Please try again.")
