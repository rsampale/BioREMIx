import streamlit as st
from functions import authenticate
from default_data import load_default_data
import pandas as pd
from rag import create_colname_vectorstore
from functions import reboot_hypothesizer
import io

# PAGE CONFIG
st.set_page_config(page_title="Data Upload")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'genes_info_df' not in st.session_state:
    st.session_state['genes_info_df'] = None
if 'genes_colmeta_dict' not in st.session_state:
    st.session_state['genes_colmeta_dict'] = None
if 'expression_df' not in st.session_state:
    st.session_state['expression_df'] = None

# check if user is authenticated
if not st.session_state['authenticated']:
    authenticate()

# Show page if user is authenticated
if st.session_state['authenticated']:

    st.sidebar.button("Reboot Session", on_click=reboot_hypothesizer,use_container_width=True,help="Completely reset the application and any changes made.")
    
    st.title("Data Upload")
    gene_data_tab, expression_data_tab, gene_list_tab = st.tabs(["Gene-annotation Data", "Gene Expression Data", "Genes of Interest (list)"])
    
    with gene_data_tab:
        st.header("Gene-annotation data upload")
        st.write("A csv where **rows = genes, and cols = metadata/information.**\n\nOptionally, you may provide a csv with two columns ('Colname', 'Description') to explainin column names.")

        with st.expander("**Upload gene information data**",expanded=True):
            st.warning("**Note:** Deleting certain columns from the default gene information data may break certain features of the app. We recommend to only modify the file by adding additional columns (annotations), or additional genes (rows)")
            ## MAIN DATAFRAME UPLOAD:
            default_allgenes_filename = "data/250219_GeneAnnotation_Data_V2.csv"
            uploaded_file = st.file_uploader("Upload your own **gene-metadata matrix**", type=["txt", "csv", "json"],key="genes_file_uploader") # not actually tested with non csv types
            
            # Read dataframe from uploaded file bytes or default file path
            if uploaded_file is not None:
                st.session_state["genes_file_bytes"] = uploaded_file.read()
                st.session_state["genes_file_name"] = uploaded_file.name

            # Use cached source or default
            if "genes_file_bytes" in st.session_state:
                raw_bytes = st.session_state["genes_file_bytes"]
                file_name = st.session_state["genes_file_name"]
                genes_df = pd.read_csv(
                    io.BytesIO(raw_bytes),
                    low_memory=False,
                    na_values=['NA', '', 'null']
                )
            else: # DEFAULT FILE READING
                with open(default_allgenes_filename, 'rb') as f:
                    raw_bytes = f.read()
                file_name = default_allgenes_filename
                genes_df = pd.read_csv(
                    default_allgenes_filename,
                    low_memory=False,
                    na_values=['NA', '', 'null']
                )

            st.write(f"**Currently Selected File Name:** {file_name}")
            st.download_button(
                label="Download Genes Data File",
                data=raw_bytes,
                file_name=file_name,
                mime="text/plain"
            )
            st.divider()
            
            ## COLUMN NAME INFORMATION DATAFRAME UPLOAD ON SIDEBAR
            default_colmeta_filename = "data/GeneData_Column_Descriptions.csv"
            uploaded_colmeta_file = st.file_uploader("Upload your own **column-name metadata matrix**", type=["txt", "csv", "json"],key="colmeta_file_uploader") # not actually tested with non csv types
            
            # Read colmeta from uploaded file bytes or default path
            if uploaded_colmeta_file is not None:
                st.session_state["colmeta_bytes"] = uploaded_colmeta_file.read()
                st.session_state["colmeta_name"] = uploaded_colmeta_file.name

            # Use cached or default colmeta source
            if "colmeta_bytes" in st.session_state:
                raw_bytes_meta = st.session_state["colmeta_bytes"]
                colmeta_file_name = st.session_state["colmeta_name"]
                colmeta_df = pd.read_csv(io.BytesIO(raw_bytes_meta))
            else: # DEFAULT
                with open(default_colmeta_filename, 'rb') as f:
                    raw_bytes_meta = f.read()
                colmeta_file_name = default_colmeta_filename
                colmeta_df = pd.read_csv(default_colmeta_filename)

            st.write(f"**Currently Selected Metadata File Name:** {colmeta_file_name}")
            st.download_button(
                label="Download Column Metadata File",
                data=raw_bytes_meta,
                file_name=colmeta_file_name,
                mime="text/plain"
            )
            
            # Process col metadata
            if "Drop.Column" in colmeta_df.columns:
                colmeta_df = colmeta_df[colmeta_df['Drop.Column'] != 'yes'] # FILTERS OUT 'reduntant' COLS AS DETERMINED BY AF
            colmeta_df['Description'] = colmeta_df['Description'].fillna(colmeta_df['Colname'])
            colmeta_dict = pd.Series(colmeta_df['Description'].values, index=colmeta_df['Colname']).to_dict()
            
            # Filter and clean genes_df
            genes_df = genes_df[list(colmeta_dict.keys())]
            genes_df.columns = genes_df.columns.str.replace('.', '_')
            colmeta_df['Colname'] = colmeta_df['Colname'].str.replace('.', '_', regex=False) # NEEDS TO BE DOWN HERE BECAUSE OTHERWISE WE GET OUT OF INDEX ERROR DUE TO TECHNICALLY WRONG NAMES
        

        # DEFINTE SESSION STATE VARIABLE FOR GENES_DF AND GENES_METADATA
        st.session_state['genes_info_df'] = genes_df
        st.session_state['genes_colmeta_df'] = colmeta_df
        st.session_state['genes_colmeta_dict'] = colmeta_dict
        
        create_colname_vectorstore() # Can only be created after genes_colmeta_df is defined
        
    ### EXPRESSION DATA SECTION ###
    
    with expression_data_tab:
        st.header("Expression Data Upload")
        st.write("A csv containing columns for genes, logFC, pvalue (adj), the disease, and the cell type used in the assay.")
        
        with st.expander("**Upload gene expression data**",expanded=True):
            ## MAIN DATAFRAME UPLOAD:
            default_expression_filename = "data/als_degs_sample_bioremix.csv"
            uploaded_expression_file = st.file_uploader("Upload your own **gene-expression data**", type=["txt", "csv", "json"],key="expr_file_uploader") # not actually tested with non csv types
            
            # Read expression data from upload or default
            if uploaded_expression_file is not None:
                st.session_state["expr_bytes"] = uploaded_expression_file.read()
                st.session_state["expr_name"] = uploaded_expression_file.name

            # Decide expression source (cached vs default)
            if "expr_bytes" in st.session_state:
                raw_expr = st.session_state["expr_bytes"]
                expr_file_name = st.session_state["expr_name"]
                expression_df = pd.read_csv(io.BytesIO(raw_expr))
            else: # DEFAULT
                with open(default_expression_filename, 'rb') as f:
                    raw_expr = f.read()
                expr_file_name = default_expression_filename
                expression_df = pd.read_csv(default_expression_filename)

            st.write(f"**Currently Selected File Name:** {expr_file_name}")
            st.download_button(
                label="Download Expression Data File",
                data=raw_expr,
                file_name=expr_file_name,
                mime="text/plain"
            )
            
        st.session_state['expression_df'] = expression_df
        
    with gene_list_tab:
        st.header("Gene List Upload")
        st.write("A txt file containing all your genes of interest, one gene symbol on each line.")
        
        with st.expander("**Upload genes of interest list**",expanded=True):
            uploaded_genelist_file = st.file_uploader("Upload your own **gene list** (see format specifications above)", type=["txt"],key="genelist_file_uploader")
            
            # Read gene list data from upload 
            if uploaded_genelist_file is not None:
                st.session_state["genelist_bytes"] = uploaded_genelist_file.read()
                st.session_state["genelist_name"] = uploaded_genelist_file.name

            if "genelist_bytes" in st.session_state: # If the user switched back onto this tab, their old upload should persist
                raw_genelist = st.session_state["genelist_bytes"]
                genelist_name = st.session_state["genelist_name"]
                lines = raw_genelist.decode('utf-8').splitlines()
                # strip whitespace and drop empties
                gene_list = [g.strip() for g in lines if g.strip()]
                
                
                st.session_state["uploaded_goi_list"] = gene_list
            
            fname = st.session_state.get("genelist_name", "No file uploaded")
            bytes_data = st.session_state.get("genelist_bytes", b"")
            has_data = "genelist_bytes" in st.session_state
                
            st.write(f"**Currently Selected File Name:** {fname}")
            st.download_button(
                label="Download Gene List Data File",
                data=bytes_data,
                file_name=fname,
                mime="text/plain",
                disabled=not has_data
            )