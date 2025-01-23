import os

import requests
import streamlit as st
from google.cloud import run_v2


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/<project>/locations/<region>"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_prompt(prompt: str):
    """Send the prompt to the backend and return the classification."""
    predict_url = f"{get_backend_url()}/predict"
    response = requests.post(predict_url, files={"prompt:": prompt}, timeout=5)
    if response.status_code == 200:
        return response.json()
    msg = f"Failed to classify prompt: {response.status_code} {response.text}"
    raise Exception(msg)


def main() -> None:
    """Main function to run the Streamlit app."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found."
        raise ValueError(msg)

    st.title("Prompt Classifier")
    st.write("Check if your answer is correct.")

    prompt = st.text_input("Prompt")
    if st.button("Classify"):
        try:
            classification = classify_prompt(prompt)
            st.write(classification)
        except Exception as e:  # noqa: BLE001
            st.error(e)
