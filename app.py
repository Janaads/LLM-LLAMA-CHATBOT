import os
import json
import streamlit as st
import pandas as pd
from groq import Groq

# Define the working directory and load the data
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

# Retrieve and set API key from config
GROQ_API_KEY = config_data.get("GROQ_API_KEY")

# Validate the API key if it exists
if not GROQ_API_KEY:
    st.error("API key is missing in the config.json file.")
    st.stop()

# Save the API key to the environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the Groq client with API key
try:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Initialize the chat history in Streamlit session state 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize the uploaded CSV data in Streamlit session state
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

# Streamlit page title
st.title("Distributed UI core Exploitation Pattern Platform using cloud")

# Direct CSV file path
csv_file_path = r"C:\Users\Jana Sorupaa\OneDrive\Desktop\chatbot\sample 1000.csv"  

# Load the CSV file directly with error handling
try:
    st.session_state.csv_data = pd.read_csv(csv_file_path, on_bad_lines='skip')
    st.success("CSV file loaded successfully!")
    # Display a preview of the CSV data
    st.dataframe(st.session_state.csv_data.head(100))
except Exception as e:
    st.error(f"Error loading CSV file: {e}")

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user message
user_prompt = st.chat_input("Ask LLAMA 3.1.....")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare context from the CSV file if loaded
    csv_context = ""
    if st.session_state.csv_data is not None:
        csv_preview = st.session_state.csv_data.head(100).to_string(index=False)
        csv_context = f"The user has uploaded a CSV file. Here is a preview of its content:\n{csv_preview}\n"

    # Send user's message to LLM with context and get a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": csv_context},
        *st.session_state.chat_history
    ]

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the LLM response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

    except Exception as e:
        st.error(f"Error while fetching the response from GROQ: {e}")
