import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


st.title("📰 Fake News Detector")
st.write(
    "Enter a news statement and let the machine learning model "
    "classify it as Fake or True."
)


news_text = st.text_area(
    "News Statement",
    placeholder="Paste a news statement here...",
    height=200
)


if st.button("Analyze News", type="primary"):

    if not news_text.strip():
        st.warning("Please enter a news statement.")
    else:

        try:
            response = requests.post(
                API_URL,
                json={"text": news_text},
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.divider()

                st.subheader("Prediction")

                if prediction == "Fake":
                    st.error("🚨 FAKE NEWS")
                else:
                    st.success("✅ TRUE NEWS")

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2%}"
                )

                st.caption(
                    "The confidence is the model's calibrated "
                    "probability estimate. It does not independently "
                    "verify the factual accuracy of the claim."
                )

            else:
                st.error(
                    f"API Error: {response.status_code}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure the API server is running."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")