import streamlit as st

from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

def create_colname_vectorstore():
    docs = []
    for _, row in st.session_state.genes_colmeta_df.iterrows():
        text = f"Column Name: {row['Colname']}\nDescription: {row['Description']}"
        doc = Document(
            page_content=text,
            metadata={
                "Colname": row['Colname'],
            }
        )
        docs.append(doc)

    st.session_state['docs'] = docs
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY, model="text-embedding-3-large")
    if "vectorstore" not in st.session_state:
        st.session_state["colname_vectorstore"] = FAISS.from_documents(docs, embeddings)

    # st.write(st.session_state.colname_vectorstore)

def col_retrieval_rag(query, rag_llm):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = st.session_state.colname_vectorstore.as_retriever(search_kwargs={"k": 10})

    # See https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/ for implementation of the new LCEL approach
    combine_docs_chain = create_stuff_documents_chain(rag_llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


    full_query = f"Briefly answer the following: What columns might be used to answer the following user query: '{query}'? Note that the user query is likely trying to perform a pandas expression / filtering step on a dataframe with rows as genes and these columns. Also mention any format info you have about those columns."
    response = rag_chain.invoke({"input": full_query})
    answer = response['answer']

    return response