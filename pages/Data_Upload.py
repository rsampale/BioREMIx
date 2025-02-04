import streamlit as st
from functions import authenticate
from default_data import load_default_data
import pandas as pd

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'genes_info_df' not in st.session_state:
    st.session_state['genes_info_df'] = None
if 'genes_colmeta' not in st.session_state:
    st.session_state['genes_colmeta_dict'] = None

# check if user is authenticated
if not st.session_state['authenticated']:
    authenticate()

# Show page if user is authenticated
if st.session_state['authenticated']:
    load_default_data()
    
    st.title("Data Upload")
    gene_data_tab, expression_data_tab = st.tabs(["Gene-annotation Data", "Gene Expression Data"])
    
    with gene_data_tab:
        st.header("Gene-annotation data upload")
        st.write("A csv where **rows = genes, and cols = metadata/information.**\n\nOptionally, you may provide a csv with two columns ('Colname', 'Description') to explainin column names.")

        with st.expander("**Upload gene information data**",expanded=True):
            ## MAIN DATAFRAME UPLOAD:
            default_allgenes_filename = "data/250129_GeneAnnotation_Data.csv"
            with open(default_allgenes_filename, 'r') as file:
                default_allgenes_content = file.read()

            uploaded_file = st.file_uploader("Upload your own **gene-metadata matrix**", type=["txt", "csv", "json"])
            if uploaded_file is not None:
                # Use uploaded file's name and content
                file_name = uploaded_file.name
                file_content = uploaded_file.read()
            else:
                # Use default file if no file is uploaded
                file_name = default_allgenes_filename
                file_content = default_allgenes_content
            # Display the file name and download button in the sidebar

            st.write(f"**Currently Selected File Name:** {file_name}")
            st.download_button(
                label="Download Genes Data File",
                data=file_content,
                file_name=file_name,
                mime="text/plain"
            )
            st.divider()
            
            ## COLUMN NAME INFORMATION DATAFRAME UPLOAD ON SIDEBAR
            default_colmeta_filename = "data/GeneData_Column_Descriptions.csv"
            with open(default_colmeta_filename, 'r') as file:
                default_colmeta_content = file.read()
        
            uploaded_colmeta_file = st.file_uploader("Upload your own **column-name metadata matrix**", type=["txt", "csv", "json"])
            if uploaded_colmeta_file is not None:
                # Use uploaded file's name and content
                colmeta_file_name = uploaded_colmeta_file.name
                colmeta_file_content = uploaded_colmeta_file.read()
            else:
                # Use default file if no file is uploaded
                colmeta_file_name = default_colmeta_filename
                colmeta_file_content = default_colmeta_content
            # Display the file name and download button in the sidebar

            st.write(f"**Currently Selected Metadata File Name:** {colmeta_file_name}")
            st.download_button(
                label="Download Column Metadata File",
                data=colmeta_file_content,
                file_name=colmeta_file_name,
                mime="text/plain"
            )
            ### ACTUALLY USE THE COLMETA - tbd
            colmeta_df = pd.read_csv(colmeta_file_name)
            if "Drop.Column" in colmeta_df.columns: # FILTERS OUT 'reduntant' COLS AS DETERMINED BY AF
                colmeta_df = colmeta_df[colmeta_df['Drop.Column'] != 'yes']
            colmeta_df['Description'] = colmeta_df['Description'].fillna(colmeta_df['Colname']) # if blank, just use the colname as the description, NOTE MIGHT BREAK IN PANDAS 3
            colmeta_dict = pd.Series(colmeta_df['Description'].values, index=colmeta_df['Colname']).to_dict()
            if 'colmeta_dict' not in st.session_state:
                st.session_state['colmeta_dict'] = colmeta_dict
            
            genes_df = pd.read_csv(file_name,low_memory=False)
            genes_df = genes_df[list(colmeta_dict.keys())] # Keep only the relevant columns, as determined by the colmeta file
            genes_df.columns = genes_df.columns.str.replace('.', '_')

        # DEFINTE SESSION STATE VARIABLE FOR GENES_DF AND GENES_METADATA
        st.session_state['genes_info_df'] = genes_df
        st.session_state['genes_colmeta_dict'] = colmeta_dict
        
        
    ### EXPRESSION DATA SECTION ###
    
    with expression_data_tab:
        st.header("Expression Data Upload")
        st.write("A csv containing columns for genes, logFC, pvalue (adj), the disease, and the cell type used in the assay.")
        
        with st.expander("**Upload gene expression data**",expanded=True):
            ## MAIN DATAFRAME UPLOAD:
            default_expression_filename = "data/als_degs_sample_bioremix.csv"
            with open(default_expression_filename, 'r') as file:
                default_expression_content = file.read()

            uploaded_expression_file = st.file_uploader("Upload your own **gene-expression data**", type=["txt", "csv", "json"]) # not actually tested with non csv types
            if uploaded_expression_file is not None:
                # Use uploaded file's name and content
                expr_file_name = uploaded_expression_file.name
                expr_file_content = uploaded_expression_file.read()
            else:
                # Use default file if no file is uploaded
                expr_file_name = default_expression_filename
                expr_file_content = default_expression_content
            # Display the file name and download button in the sidebar

            st.write(f"**Currently Selected File Name:** {expr_file_name}")
            st.download_button(
                label="Download Expression Data File",
                data=expr_file_content,
                file_name=expr_file_name,
                mime="text/plain"
            )
            
            expression_df = pd.read_csv(expr_file_name)
            
        st.session_state['expression_df'] = expression_df