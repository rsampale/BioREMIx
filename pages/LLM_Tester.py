import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate 
from openai import OpenAI
from functions import *
from pathlib import Path


# check if authenticated is in session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# check if user is authenticated
if not st.session_state['authenticated']:
    authenticate()

# Show page if user is authenticated
if st.session_state['authenticated']:

    # Get API key
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    # Check if key was retrieved 
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
                    


    # set up memory
    msgs = StreamlitChatMessageHistory(key = "langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)

    # provide an initial message
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hi! How can I help you?")

    # template for prompt 
    health_template = """ 
    Task: Respond to the users questions as a pirate. Responses should not lose information, but should be spoken using 'pirate-speech'
    Format: markdown; **include ```LLM Response``` headings**

    {history}
    User: {human_input}
    GPT: 
    """

    # prompt the llm and send 
    prompt = PromptTemplate(input_variables=["history", "human_input"], template= health_template)
    llm_chain = LLMChain(llm=ChatOpenAI(openai_api_key = OPENAI_API_KEY, model = "gpt-4o"), prompt=prompt, memory=memory)


    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.spinner("Generating response..."):
            response = llm_chain.run(prompt)
        st.session_state.last_response = response
        st.chat_message("assistant").write(response)
               

                
    # clear chat button
    clear_memory = st.sidebar.button("Clear Chat")
    if clear_memory:
        clear_session_state_except_password()
        st.session_state["last_response"] = "GPT: Hi there, hope you're doing great! How can I help you?"