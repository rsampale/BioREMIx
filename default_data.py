import streamlit as st
import pandas as pd


def load_default_data():
    if 'genes_info_df' not in st.session_state:
        st.session_state['genes_info_df'] = None
    if 'genes_colmeta' not in st.session_state:
        st.session_state['genes_colmeta_dict'] = None

    try:
        default_allgenes_filename = "data/241016_DiseaseGene_Localization.csv"
        with open(default_allgenes_filename, 'r') as file:
            default_allgenes_content = file.read()
        file_name = default_allgenes_filename
        file_content = default_allgenes_content
        

        default_colmeta_filename = "data/240814_DiseaseGene_colmetadata_AFannotated.csv"
        with open(default_colmeta_filename, 'r') as file:
            default_colmeta_content = file.read()
        colmeta_file_name = default_colmeta_filename
        colmeta_file_content = default_colmeta_content


        # Build the datafranes and dict from the files:

        colmeta_df = pd.read_csv(colmeta_file_name)
        if "Drop.Column" in colmeta_df.columns: # FILTERS OUT 'reduntant' COLS AS DETERMINED BY AF
            colmeta_df = colmeta_df[colmeta_df['Drop.Column'] != 'yes']
        colmeta_df['Description'] = colmeta_df['Description'].fillna(colmeta_df['Colname']) # if blank, just use the colname as the description, NOTE MIGHT BREAK IN PANDAS 3
        colmeta_dict = pd.Series(colmeta_df['Description'].values, index=colmeta_df['Colname']).to_dict()
        if 'colmeta_dict' not in st.session_state:
            st.session_state['colmeta_dict'] = colmeta_dict
        
        genes_df = pd.read_csv(file_name)
        genes_df = genes_df[list(colmeta_dict.keys())] # Keep only the relevant columns, as determined by the colmeta file
        genes_df.columns = genes_df.columns.str.replace('.', '_')

        # DEFINTE SESSION STATE VARIABLES
        st.session_state['genes_info_df'] = genes_df
        st.session_state['genes_colmeta_dict'] = colmeta_dict
    except:
        print("Error loading the default genes/info data.")