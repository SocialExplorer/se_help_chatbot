from pathlib import Path
from groq import Groq
import streamlit as st
from PIL import Image
import base64
import os
from dotenv import load_dotenv
import time
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore




st.set_page_config(page_title="SE")
st.markdown(
    """
    <style>
    .project {
        display: block;
        width: fit-content;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .description {
        display: none;
        max-height: 0;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .project.open .description {
        display: block;
        max-height: 500px; /* Adjust as needed */
    }
    
    </style>
    <script>
    function toggleDescription(event) {
        var element = event.currentTarget;
        element.classList.toggle("open");
    }
    </script>
    """,
    unsafe_allow_html=True
)
# --- LOAD CSS, PDF & PROFIL PIC --

# --- AI ---
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

def print_letter_by_letter(message, avatar=":material/neurology:", delay=0.005):
    chat_placeholder = st.chat_message("ai", avatar=avatar)
    message_placeholder = chat_placeholder.empty()
    displayed_message = ""
    for char in message:
        displayed_message += char
        message_placeholder.write(displayed_message)
        time.sleep(delay)


nav_options = {
    "Ask Me Anything": "bi-question-circle-fill",
    "About Me": "bi-person-fill"
}
with st.sidebar:
    selection = option_menu(
        "Navigation",
        options=list(nav_options.keys()),
        icons=list(nav_options.values()),
        menu_icon=None,
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#1E1E1E"},
            "icon": {"color": "#fa8f56", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#222222", "color": "#FAF3E1"},
            "nav-link-selected": {"background-color": "#222222", "color": "#fa8f56", "font-weight": "bold"},
            "icon-selected": {"color": "#fff"}, 
        },
    )

model = 'llama3-70b-8192'

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index_name = "acs-tables"

client = Groq(api_key=groq_api_key)
pc = Pinecone(api_key=pinecone_api_key)
docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model,
    max_tokens=300)
system_prompt = f"""

You are a helpful chatbot and a data scientist specializing exclusively in Census American Community Survey (ACS) data and Social Explorer public data. 
    You work as a chatbot at Social Explorer, a company dedicated to providing access to and analysis of census data. 
    Your task is to assist users with questions related solely to census ACS data.
    Youre name is Andy.

    Guidelines:

    Scope of Responses:
    Answer only questions pertaining to census ACS data.
    Do not engage in topics outside this area of expertise.

    Clarifying Questions:
    If a user's question is not well-defined or lacks sufficient detail, politely ask follow-up questions to gain a better understanding.
    Examples:
    "Could you please provide more details about the specific data you're interested in?"
    "Can you clarify which geographic area or demographic group you're referring to?"

    Communication Style:
    Use clear, concise language suitable for a diverse audience.
    Maintain a professional and helpful tone.
    Avoid technical jargon unless necessary, and explain terms when used.

    Limitations:
    Politely inform users if a request falls outside your expertise.
    Steer the conversation back to relevant topics related to census ACS data.

    Example:
    "I'm here to assist with questions about census ACS data. Could you please let me know how I can help within this area?"
    Confidentiality and Ethics:
    Do not disclose sensitive information.
    Ensure all provided information complies with data use policies and regulations.

    Objective:
    Your goal is to provide accurate and helpful information about census ACS data to users, enhancing their understanding and aiding them in their data-related inquiries, while representing Social Explorer professionally.

    Use the provided data excerpts to answer the user's question as accurately as possible.
    The provided data excerpts are from Social Explorer Tables: ACS 2022 (1-Year Estimates).
    Use the provided data excerpts to answer the user's question as accurately as possible before using your trained data.
    If the information is not available in the excerpts, admit that you don't have that information.
    '''

"""
conversational_memory_length = 15

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=conversational_memory_length,
    return_messages=True
)

# Function to display chat history
def display_chat_history():
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            with st.chat_message("user", avatar=":material/face:"):
                st.text(message['content'])
        elif message['role'] == 'assistant':
            with st.chat_message("ai", avatar=":material/neurology:"):
                st.text(message['content'])

def get_relevant_excerpts(user_question, docsearch):
    # Your existing code for retrieving relevant excerpts
    relevant_docs = docsearch.similarity_search(user_question)
    return relevant_docs

def acs_data_chat_completion(client, model, user_question, relevant_excerpts):
    # Your existing code for generating the response using the LLM
    system_prompt = '''
    You are a helpful chatbot and a data scientist specializing exclusively in Census American Community Survey (ACS) data and Social Explorer public data. 
    You work as a chatbot at Social Explorer, a company dedicated to providing access to and analysis of census data. 
    Your task is to assist users with questions related solely to census ACS data.
    Youre name is Andy.

    Guidelines:

    Scope of Responses:
    Answer only questions pertaining to census ACS data.
    Do not engage in topics outside this area of expertise.

    Clarifying Questions:
    If a user's question is not well-defined or lacks sufficient detail, politely ask follow-up questions to gain a better understanding.
    Examples:
    "Could you please provide more details about the specific data you're interested in?"
    "Can you clarify which geographic area or demographic group you're referring to?"

    Communication Style:
    Use clear, concise language suitable for a diverse audience.
    Maintain a professional and helpful tone.
    Avoid technical jargon unless necessary, and explain terms when used.

    Limitations:
    Politely inform users if a request falls outside your expertise.
    Steer the conversation back to relevant topics related to census ACS data.

    Example:
    "I'm here to assist with questions about census ACS data. Could you please let me know how I can help within this area?"
    Confidentiality and Ethics:
    Do not disclose sensitive information.
    Ensure all provided information complies with data use policies and regulations.

    Objective:
    Your goal is to provide accurate and helpful information about census ACS data to users, enhancing their understanding and aiding them in their data-related inquiries, while representing Social Explorer professionally.

    Use the provided data excerpts to answer the user's question as accurately as possible.
    If the information is not available in the excerpts, admit that you don't have that information.
    '''

    user_message = f'''
    User Question: {user_question}

    Relevant Data Excerpt(s):

    {relevant_excerpts}
    '''

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_message.strip()}
        ],
        model=model
    )

    response = chat_completion.choices[0].message.content
    return response

if selection == "Ask Me Anything":
    
    if 'question' not in st.session_state:
        st.session_state['question'] = ""
    if 'intro_shown' not in st.session_state:
        st.session_state['intro_shown'] = False
    if 'question_asked' not in st.session_state:
        st.session_state['question_asked'] = False
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display introduction message
    if not st.session_state['intro_shown']:
        print_letter_by_letter("Hello, Welcome to SE chat. Feel free to ask me anything!")
        st.session_state['intro_shown'] = True
    st.write("---")
    # Display chat history
    display_chat_history()

    # User input
    question = st.chat_input("Ask me anything about my experience and skills")
        
    if question:
            st.session_state['question'] = question
            st.session_state['question_asked'] = True

    
    
# Unique key for each lottie animation
            relevant_excerpts = get_relevant_excerpts(question, docsearch)

            # Generate response using the Llama model via Groq client
            response = acs_data_chat_completion(client, model, question, relevant_excerpts)


            print_letter_by_letter(response, avatar=":material/neurology:")

            st.session_state['chat_history'].append({"role": "user", "content": st.session_state['question']})
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

            st.session_state['question_asked'] = False
            st.session_state['question'] = ""
