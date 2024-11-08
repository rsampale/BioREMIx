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
 
    st.sidebar.button("Reboot Session", on_click=clear_session_state_except_password,use_container_width=True)



    # Initialize csv/dataframe being searched and manipulated - display it in the sidebar at all times for download and info
    if 'refine_section_visible' not in st.session_state:
        st.session_state['refine_section_visible'] = True

    ## MAIN DATAFRAME UPLOAD ON SIDEBAR:
    default_allgenes_filename = "data/241016_DiseaseGene_Localization.csv"
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
    colmeta_df = pd.read_csv(colmeta_file_name)
    colmeta_df['Description'].fillna(colmeta_df['Colname'], inplace=True) # if blank, just use the colname as the description
    colmeta_dict = pd.Series(colmeta_df['Description'].values, index=colmeta_df['Colname']).to_dict()
    if 'colmeta_dict' not in st.session_state:
        st.session_state['colmeta_dict'] = colmeta_dict
    
    genes_df = pd.read_csv(file_name)
    genes_df.columns = genes_df.columns.str.replace('.', '_')

    if 'user_researchquestion' not in st.session_state:
        st.session_state['user_researchquestion'] = None
    if 'relevant_cols_only_df' not in st.session_state:
        st.session_state['relevant_cols_only_df'] = None
    if 'user_refined_df' not in st.session_state:
        st.session_state['user_refined_df'] = None
    if 'user_refinement_q' not in st.session_state:
        st.session_state['user_refinement_q'] = None

    llm = ChatOpenAI(temperature=0, model='gpt-4o', openai_api_key=OPENAI_API_KEY)

    # PAGE FORMAT CODE START
    # Make into invisible container so it can be hidden with appropriate buttons?
    if 'do_refine_loop' not in st.session_state:
        st.session_state['do_refine_loop'] = None
        
    if st.session_state['refine_section_visible']:
        st.title("Hypothesis Formulation Tool")
        st.divider()

        if not st.session_state['user_researchquestion']:
            # FIND COLUMNS RELEVANT TO HYPOTHESIS - MAYBE ADD AS AN OPTION LATER (AND KEEP ALL BY DEFAULT)
            st.header("Develop your search space",divider='green')
            st.subheader("What are you interested in exploring today?")
            st.write("**This will subset your data to include only relevant columns.**")
            rq_box = st.container(height=150)
            with rq_box:
                user_researchquestion = st.text_input("E.g. 'I want to analyze enzymatic proteins that are associated with ALS'",max_chars=500)
                if user_researchquestion:
                    st.session_state['user_researchquestion'] = user_researchquestion
                st.write("**Your Question:** ",st.session_state['user_researchquestion'])
                # FOR COPY PASTE: I am interested in finding out common cytosolic genes implicated in both Parkinsons and ALS
            if not st.session_state['user_researchquestion']:
                st.markdown("Please provide a research quesiton/hypothesis to proceed.")

        if st.session_state['user_researchquestion']:
            possible_columns = list(genes_df.columns)
            # set up prompt:
            prompt = PromptTemplate(
            template=
            "Here is a list of column names in a dataframe: {col_names}"
            "The columns hold information relating to gene names, disease associations, biological processes, and much more"
            "Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset"
            "Consult the following dictionary to understand the meanings of columns you initially do not understand: {colmeta_dict}\n"
            "Here is the user's research question or hypothesis: {query}"
            "Using this query and the list of column names, select any column names you think might be relevant to their question or future analysis"
            "Return the column names relevant to the query in a list format. Remember, it is better to give more columns than necessary than to give not enough."
            "To do this, it may actually be easier to think which to exclude because they are most likely irrelevant. For example, you may almost always"
            "want to include localization or expression columns because those can be very useful for answering future questions the user may have."
            "Format instructions: Return ONLY a list in the format 'col1,col2,col3' (without the quotes, and with no brackets or anything)"
            )
    
            chain = prompt | llm 
            parser_output = chain.invoke({"query": st.session_state['user_researchquestion'], "col_names": possible_columns,"colmeta_dict": colmeta_dict})
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
                pd_df_agent.handle_parsing_errors = "Check your output and make sure it conforms, use the Action Input/Final Answer syntax"
                full_prompt = st.session_state['user_refinement_q'] + ". Your response should just be the pandas expression required to achieve this. Do not include code formatting markers like backticks. E.g. you might return df_y = dfx[dfx['blah'] == 'foo']"
                response = pd_df_agent.run(full_prompt)
                # response = pd_df_agent.run(st.session_state['user_refinement_q'])
                # st.write(response) # FOR DEBUGGING LLM OUTPUT
                pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
                pandas_code_only = pandas_code_only.replace("df", "relevant_cols_only_df")
                pandas_code_only = pandas_code_only.rstrip('`') # remove code backticks left over
                # st.write(f"Code to be evaluated:{pandas_code_only}") # FOR DEBUGGING LLM OUTPUT
                user_refined_df = eval(pandas_code_only)
                st.session_state['user_refined_df'] = user_refined_df
                st.dataframe(user_refined_df)
                
                
                
                # ADD 3 BUTTONS FOR: RE-REFINE, CHAT WITH DATA, ANALYZE
                left_col, middle_col, right_col = st.columns(3)
                
                left_col.button("Keep Refining", icon="🔃", use_container_width=True,on_click=refineloop_buttonclick)
                middle_col.button("Chat with your Data", icon="💬", use_container_width=True,on_click=chat_buttonclick)
                right_col.button("Ready To Analyze", icon="🔬", use_container_width=True,on_click=analyze_buttonclick)
                    
    else: # refine_section_visible is FALSE
        
        if 'show_chat_analyze_buttons' not in st.session_state:
            st.session_state['show_chat_analyze_buttons'] = True
        if 'show_refine_analyze_buttons' not in st.session_state:
            st.session_state['show_refine_analyze_buttons'] = True
        if 'data_chat' not in st.session_state:
            st.session_state['data_chat'] = False
        if 'analyze_data' not in st.session_state:
            st.session_state['analyze_data'] = False

        ### REPEAT REFINEMENT: 
        
        if st.session_state['do_refine_loop']:
            repeat_refinement(llm=llm)

        if st.session_state['data_chat']:
            chat_with_data(llm=llm)

        if st.session_state['analyze_data']:
            analyze_data(llm=llm)

        left_col, right_col = st.columns(2)
        chat_button_ph = st.empty()
        analyze_button_ph = st.empty()

        if st.session_state['show_chat_analyze_buttons']:
            left_col.button("Chat with your Data", icon="💬", use_container_width=True,on_click=chat_buttonclick)
            right_col.button("Ready To Analyze", icon="🔬", use_container_width=True,on_click=analyze_buttonclick)

        if st.session_state['show_refine_analyze_buttons']:
            left_col.button("Keep Refining", icon="🔃", use_container_width=True,on_click=refineloop_buttonclick)
            right_col.button("Ready To Analyze", icon="🔬", use_container_width=True,on_click=analyze_buttonclick)
        
