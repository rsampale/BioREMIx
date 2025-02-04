import streamlit as st
import pandas as pd

# imports for vectorstore/rag:
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import retrieval_qa
from langchain import hub
from default_data import create_colname_vectorstore, load_default_data

load_default_data()
create_colname_vectorstore()



llm = ChatOpenAI(openai_api_key = st.secrets.OPENAI_API_KEY, model = "gpt-4o")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = st.session_state.colname_vectorstore.as_retriever(search_kwargs={"k": 10})

# See https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/ for implementation of the new LCEL approach
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# st.write(st.session_state['docs'])

query = st.text_input("Enter your question:")
if query:
    response = rag_chain.invoke({"input": query})

    # Display the final answer
    st.subheader("LLM Answer:")
    st.write(response['answer'])

    # Display retrieved documents separately
    st.subheader("Retrieved Documents:")
    for doc in response['context']:
        st.write(f"**Content:** {doc.page_content}")
        st.write("---")