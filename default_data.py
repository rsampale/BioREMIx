import streamlit as st
import pandas as pd

# imports for vectorstore/rag:
from rag import create_colname_vectorstore


def load_default_data():
    if 'genes_info_df' not in st.session_state:
        st.session_state['genes_info_df'] = None
    if 'genes_colmeta' not in st.session_state:
        st.session_state['genes_colmeta_dict'] = None
    if 'expression_df' not in st.session_state:
        st.session_state['expression_df'] = None

    try:
        default_allgenes_filename = "data/250129_GeneAnnotation_Data.csv"
        with open(default_allgenes_filename, 'r') as file:
            default_allgenes_content = file.read()
        file_name = default_allgenes_filename
        file_content = default_allgenes_content
        

        default_colmeta_filename = "data/GeneData_Column_Descriptions.csv"
        with open(default_colmeta_filename, 'r') as file:
            default_colmeta_content = file.read()
        colmeta_file_name = default_colmeta_filename
        colmeta_file_content = default_colmeta_content
        
        default_expression_filename = "data/als_degs_sample_bioremix.csv"
        with open(default_expression_filename, 'r') as file:
            default_expression_content = file.read()
        expression_file_name = default_expression_filename
        expression_file_content = default_expression_content


        # Build the datafranes and dict from the files:

        colmeta_df = pd.read_csv(colmeta_file_name)
        if "Drop.Column" in colmeta_df.columns: # FILTERS OUT 'reduntant' COLS AS DETERMINED BY AF
            colmeta_df = colmeta_df[colmeta_df['Drop.Column'] != 'yes']
        colmeta_df['Description'] = colmeta_df['Description'].fillna(colmeta_df['Colname']) # if blank, just use the colname as the description, NOTE MIGHT BREAK IN PANDAS 3
        colmeta_dict = pd.Series(colmeta_df['Description'].values, index=colmeta_df['Colname']).to_dict()
        
        genes_df = pd.read_csv(file_name,low_memory=False)
        genes_df = genes_df[list(colmeta_dict.keys())] # Keep only the relevant columns, as determined by the colmeta file
        genes_df.columns = genes_df.columns.str.replace('.', '_')
        colmeta_df['Colname'] = colmeta_df['Colname'].str.replace('.', '_', regex=False)
        
        expression_df = pd.read_csv(expression_file_name)

        # DEFINTE SESSION STATE VARIABLES
        st.session_state['genes_info_df'] = genes_df
        st.session_state['genes_colmeta_dict'] = colmeta_dict
        st.session_state['genes_colmeta_df'] = colmeta_df
        st.session_state['expression_df'] = expression_df
        create_colname_vectorstore()
    except:
        print("Error loading the default genes/info data.")