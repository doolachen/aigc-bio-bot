import os  


import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

API_URL = "http://localhost:8099/v1"
MODEL_NAME = "models/DeepSeek-R1-Distill-Qwen-32B-Lora-Full-4bit"

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    openai_api_base=API_URL,  
    openai_api_key="sk-fake-key",  # fake key
    temperature=0.1
)

SYSTEM_PROMPT = """
Below is a task description paired with additional context. Your assignment is to craft a response that fully addresses the request. 
Prior to providing your final answer, carefully analyze the question and outline a step-by-step reasoning process (chain of thought) to ensure that your final response is both logical and precise.

### Instruction:
You are a distinguished medical professional with extensive expertise in clinical diagnostics, patient assessment, and treatment strategy formulation. Your task is to answer the following medical inquiry.

"""

st.title("BioMedical Reasoning Bot\n By Lingyi Zhang")
st.write("Model: DeepSeek-R1-Distill-Qwen-32B(4-bit quantization) with Lora fine-tune")


def chat_stream(user_question):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"### Question:\n{user_question}\n\n### Response:\n<think>")
    ]
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "history" not in st.session_state:
    st.session_state.history = []

for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):  
        st.write(message["content"])  
        if message["role"] == "assistant":  
            feedback = message.get("feedback", None)  
            st.session_state[f"feedback_{i}"] = feedback

if prompt := st.chat_input("Input your question"): 
    with st.chat_message("user"):  
        st.write(prompt)

    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        full_response = ""

        for chunk in st.write_stream(chat_stream(prompt)):
            full_response += chunk

        st.feedback(
            "thumbs",
            key=f"feedback_{len(st.session_state.history)}",
            on_change=save_feedback,
            args=[len(st.session_state.history)],
        )
    st.session_state.history.append({"role": "assistant", "content": full_response})