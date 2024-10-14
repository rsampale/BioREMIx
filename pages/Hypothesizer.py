import streamlit as st
from functions import *
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

mgi_icon = "images/the_elizabeth_h_and_james_s_mcdonnell_genome_institute_logo.jpg"
st.logo(mgi_icon, size='large')

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
 
    st.sidebar.button("Reboot Session", on_click=clear_session_state_except_password(),use_container_width=True)



    # Initialize csv/dataframe being searched and manipulated - display it in the sidebar at all times for download and info
    
    ## MAIN DATAFRAME UPLOAD ON SIDEBAR:
    default_allgenes_filename = "data/240814_DiseaseGene_Localization.csv"
    with open(default_allgenes_filename, 'r') as file:
        default_allgenes_content = file.read()
    uploaded_file = st.sidebar.file_uploader("Upload your own **gene-metadata matrix**", type=["txt", "csv", "json"])
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
        label="Download Genes Data File",
        data=file_content,
        file_name=file_name,
        mime="text/plain"
    )
    st.sidebar.divider()
    
    ## COLUMN NAME INFORMATION DATAFRAME UPLOAD ON SIDEBAR
    default_colmeta_filename = "data/240814_DiseaseGene_colmetadata.csv"
    with open(default_colmeta_filename, 'r') as file:
        default_colmeta_content = file.read()
    uploaded_colmeta_file = st.sidebar.file_uploader("Upload your own **column-name metadata matrix**", type=["txt", "csv", "json"])
    if uploaded_colmeta_file is not None:
        # Use uploaded file's name and content
        colmeta_file_name = uploaded_colmeta_file.name
        colmeta_file_content = uploaded_colmeta_file.read()
    else:
        # Use default file if no file is uploaded
        colmeta_file_name = default_colmeta_filename
        colmeta_file_content = default_colmeta_content
    # Display the file name and download button in the sidebar
    st.sidebar.write(f"**Currently Selected Metadata File Name:** {colmeta_file_name}")
    st.sidebar.download_button(
        label="Download Column Metadata File",
        data=colmeta_file_content,
        file_name=colmeta_file_name,
        mime="text/plain"
    )
    ### ACTUALLY USE THE COLMETA - tbd 
    
    genes_df = pd.read_csv(file_name)
    genes_df.columns = genes_df.columns.str.replace('.', '_')

    if 'user_researchquestion' not in st.session_state:
        st.session_state['user_researchquestion'] = None
    if 'relevant_cols_only_df' not in st.session_state:
        st.session_state['relevant_cols_only_df'] = None
    if 'user_refinement_q' not in st.session_state:
        st.session_state['user_refinement_q'] = None

    llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=OPENAI_API_KEY)

    # PAGE FORMAT CODE START
    st.title("Hypothesis Formulation Tool")
    st.divider()
    
    if not st.session_state['user_researchquestion']:
        # FIND COLUMNS RELEVANT TO HYPOTHESIS - MAYBE ADD AS AN OPTION LATER (AND KEEP ALL BY DEFAULT)
        st.header("Develop your search space",divider='green')
        st.subheader("What are you interested in exploring today?")
        rq_box = st.container(height=150)
        with rq_box:
            user_researchquestion = st.text_input("E.g. 'I want to find ATP binding enzyme's that are associated with Alzheimer's'",max_chars=500)
            if user_researchquestion:
                st.session_state['user_researchquestion'] = user_researchquestion
            st.write("**Your Question:** ",st.session_state['user_researchquestion'])
            # FOR COPY PASTE: I am interested in finding out common cytosolic genes implicated in both Parkinsons and ALS
        st.markdown("Please provide a research quesiton/hypothesis to proceed.")

    if st.session_state['user_researchquestion']:
        possible_columns = list(genes_df.columns)
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
        "Format instructions: Return ONLY a list in the format 'col1,col2,col3' (without the quotes, and with no brackets or anything)"
        )
  
        chain = prompt | llm 
        parser_output = chain.invoke({"query": st.session_state['user_researchquestion'], "col_names": possible_columns})
        # st.write(parser_output)
        colnames_list = parser_output.content.split(",")
        # st.write(colnames_list)
        relevant_cols_only_df = genes_df[colnames_list]
        st.session_state['relevant_cols_only_df'] = relevant_cols_only_df
        st.dataframe(st.session_state['relevant_cols_only_df'])

        st.divider()
        st.header("Refine your results",divider='green')
        refine_box = st.container(height=150)
        with refine_box:
            user_refinement_q = st.text_input("E.g. 'Only keep genes with low tissue specificity'",max_chars=500)
            if user_refinement_q:
                st.session_state['user_refinement_q'] = user_refinement_q
            st.write("**Your Data Refinement Query:** ",st.session_state['user_refinement_q'])

        if st.session_state['user_refinement_q']:
            additional_prefix = """
                Instructions: A user will give you a refining statement. By looking at the column names, rows, and cell values in the dataframe provided,
                you will try and filter the dataframe to match their specifications. Pay attention to this specific dataframe
                and make sure to use column names etc. that are real and relevant. User specification:\n
            """
            additional_suffix = """
                \n Additional instructions: ONLY Give me the pandas operation/code to retrieve the relevant columns
                with no additional quotes or formatting (i.e. backticks for code blocks).
            """
            pd_df_agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state['relevant_cols_only_df'],
                # agent_type="tool-calling", # can also be others like 'openai-tools' or 'openai-functions'
                verbose=True,
                allow_dangerous_code=True,
                # prefix=additional_prefix,
                # suffix=additional_suffix, # AS SOON AS YOU ADD A SUFFIX IT GETS CONFUSED ABOUT THE ACTUAL COL NAMES. DOES NOT SEEM TO BE IN THE SAME CONTEXT. 
                include_df_in_prompt=True,
                number_of_head_rows=10
            )
            pd_df_agent.handle_parsing_errors = True
            full_prompt = st.session_state['user_refinement_q'] + ". Your response should just be the pandas expression required to achieve this. Do not include code formatting markers like backticks. E.g. you might return df_y = dfx[dfx['blah'] == 'foo']"
            response = pd_df_agent.run(full_prompt)
            # response = pd_df_agent.run(st.session_state['user_refinement_q'])
            st.write(response)
            pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
            pandas_code_only = pandas_code_only.replace("df", "relevant_cols_only_df")
            pandas_code_only = pandas_code_only.rstrip('`') # remove code backticks left over
            st.write(f"Code to be evaluated:{pandas_code_only}")
            user_refined_df = eval(pandas_code_only)
            st.dataframe(user_refined_df)
            
            
            
            # ADD 3 BUTTONS FOR: RE-REFINE, CHAT WITH DATA, ANALYZE
            left_col, middle_col, right_col = st.columns(3)
            
            if left_col.button("Keep Refining", icon="🔃", use_container_width=True):
                left_col.markdown("You clicked the refining button.")
            if middle_col.button("Chat with your Data", icon="💬", use_container_width=True):
                middle_col.markdown("You clicked the chat button.")
            if right_col.button("Ready To Analyze", icon="🔬", use_container_width=True):
                right_col.markdown("You clicked the analyze button.")
            
        # user_refined_df = pandas_str

        
