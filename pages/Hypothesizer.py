import streamlit as st
from functions import *
import pprint
from typing import Any, Dict
import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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
    # set up memory?
 




    # Initialize csv/dataframe being searched and manipulated - display it in the sidebar at all times for download and info
    default_allgenes_filename = "data/240814_DiseaseGene_Localization.csv"
    with open(default_allgenes_filename, 'r') as file:
        default_allgenes_content = file.read()
    uploaded_file = st.sidebar.file_uploader("Upload your own gene-metadata matrix", type=["txt", "csv", "json"])
    if uploaded_file is not None:
        # Use uploaded file's name and content
        file_name = uploaded_file.name
        file_content = uploaded_file.read()
    else:
        # Use default file if no file is uploaded
        file_name = default_allgenes_filename
        file_content = default_allgenes_content
    # Display the file name and download button in the sidebar
    st.sidebar.write(f"**Currently Selected File Name:** {file_name}")
    st.sidebar.download_button(
        label="Download File",
        data=file_content,
        file_name=file_name,
        mime="text/plain"
    )
    genes_df = pd.read_csv(file_name)
    genes_df.columns = genes_df.columns.str.replace('.', ' ')


    st.title("Hypothesis Formulation Tool")
    st.divider()
    st.header("Develop your search space",divider='green')
    st.subheader("What are you interested in exploring today?")
    rq_box = st.container(height=150)
    with rq_box:
        user_researchquestion = st.chat_input("E.g. 'What mitochondria-specific genes are implicated in Alzheimer's?'",max_chars=500)
        st.write("**Your Question:** ",user_researchquestion)

    # FIND COLUMNS RELEVANT TO HYPOTHESIS - MAYBE ADD AS AN OPTION LATER (AND KEEP ALL BY DEFAULT)
    if user_researchquestion:
        possible_columns = list(genes_df.columns)
        llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=OPENAI_API_KEY)
        # set up prompt:
        prompt = PromptTemplate(
        template=
        "Here is a list of column names in a dataframe: {col_names}"
        "The columns hold information relating to gene names, disease associations, biological processes, and much more"
        "Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset"
        "Explanations of some acronyms: HPA = Human protein atlas, "
        "Here is the user's research question or hypothesis: {query}"
        "Using this query and the list of column names, select any column names you think might be relevant to their question or future analysis"
        "Return the column names relevant to the query in a list format. Remember, it is better to give more columns than necessary than to give not enough."
        "Format instructions: Return ONLY a list in the format col1,col2,col3,etc."
        )
  
        chain = prompt | llm 
        parser_output = chain.invoke({"query": user_researchquestion, "col_names": possible_columns})
        # st.write(parser_output)
        colnames_list = parser_output.content.split(",")
        relevant_cols_only_df = genes_df[colnames_list]
        st.dataframe(genes_df[colnames_list])

        st.divider()
        st.header("Refine your results",divider='green')
        
