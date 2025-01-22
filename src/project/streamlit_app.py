import requests
import streamlit as st

# Set the title of the Streamlit app
st.title("Multiple Choice Prediction")

# Input for the query
query = st.text_input("Enter your query:")

# Input for the choices
choice1 = st.text_input("Enter your choice 1:")
choice2 = st.text_input("Enter your choice 2:")
choice3 = st.text_input("Enter your choice 3:")
choice4 = st.text_input("Enter your choice 4:")

# Button to submit the request
if st.button("Predict"):
    if query and choice1 and choice2 and choice3 and choice4:
        # Prepare the choices list
        choices_list = [choice1, choice2, choice3, choice4]

        # Prepare the request payload
        payload = {"query": query, "choices": choices_list}

        # Send the request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            q = f"Query: {query}"
            st.write(q)
            st.write(result["choices"][0], round(result["predictions"][0], 3) * 100, "%")
            st.write(result["choices"][1], round(result["predictions"][1], 3) * 100, "%")
            st.write(result["choices"][2], round(result["predictions"][2], 3) * 100, "%")
            st.write(result["choices"][3], round(result["predictions"][3], 3) * 100, "%")
        else:
            st.error("Error: " + response.text)
    else:
        st.warning("Please fill in both the query and choices.")
