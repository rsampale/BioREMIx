import streamlit as st
from default_data import load_default_data
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
    
    # Load the genes_info default data
    load_default_data()
    
    # Get API keys
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key is not set. Please set the PERPLEXITY_API_KEY environment variable.")
 
    st.sidebar.button("Reboot Session", on_click=reboot_hypothesizer,use_container_width=True)


    if 'refine_section_visible' not in st.session_state:
        st.session_state['refine_section_visible'] = True
    if 'user_researchquestion' not in st.session_state:
        st.session_state['user_researchquestion'] = None
    if 'relevant_cols_only_df' not in st.session_state:
        st.session_state['relevant_cols_only_df'] = None
    if 'user_refined_df' not in st.session_state:
        st.session_state['user_refined_df'] = None
    if 'user_refinement_q' not in st.session_state:
        st.session_state['user_refinement_q'] = None
    if 'skipped_col_filter' not in st.session_state:
        st.session_state['skipped_col_filter'] = False
    if 'skipped_initial_refine' not in st.session_state:
        st.session_state["skipped_initial_refine"] = False

    # Text elements
    if 'provide_rq_text' not in st.session_state:
        st.session_state['provide_rq_text'] = "Please provide a research question/hypothesis to proceed."
    if 'provide_refinement_text' not in st.session_state:
        st.session_state['provide_refinement_text'] = "Please provide a refining statement to proceed."

    # CREATE MODELS
    llm_4o = ChatOpenAI(temperature=0, model='gpt-4o-2024-11-20', openai_api_key=OPENAI_API_KEY)
    llm_o1mini = ChatOpenAI(temperature=1,model='o1-mini', openai_api_key=OPENAI_API_KEY) # Actually might work now, but takes a long time. Investigate further by printing response.
    llm_o1prev = ChatOpenAI(temperature=1,model='o1-preview', openai_api_key=OPENAI_API_KEY)

    # PAGE FORMAT CODE START
    # Make into invisible container so it can be hidden with appropriate buttons?
    if 'do_refine_loop' not in st.session_state:
        st.session_state['do_refine_loop'] = None
        
    if st.session_state['refine_section_visible']:
        st.title("Hypothesis Formulation Tool")
        st.divider()

        # FIND COLUMNS RELEVANT TO HYPOTHESIS - MAYBE ADD AS AN OPTION LATER (AND KEEP ALL BY DEFAULT)
        st.header("Develop your search space",divider='green')
        st.subheader("What are you interested in exploring today?")
        st.write("**This will subset your data to include only relevant columns.**")
        rq_box = st.container(height=150)
        with rq_box:
            user_researchquestion = st.text_input("E.g. 'I want to analyze enzymatic proteins that are associated with ALS'",max_chars=500)
            if user_researchquestion:
                st.session_state['user_researchquestion'] = user_researchquestion
                st.session_state["skipped_col_filter"] = False
            st.write("**Your Question:** ",st.session_state['user_researchquestion'])
            # FOR COPY PASTE: I am interested in finding out common cytosolic genes implicated in both Parkinsons and ALS
        if not st.session_state['user_researchquestion']:
            col1, col2 = st.columns([0.7,0.3])
            with col1:
                if not st.session_state['skipped_col_filter']:
                    st.markdown(st.session_state.provide_rq_text)
            with col2:
                if st.button("Keep all columns", use_container_width=True, on_click=clear_text(text_element='rq_prompt')):
                    st.session_state['relevant_cols_only_df'] = st.session_state['genes_info_df'] # if they want all columns, the relevant ones are just all
                    st.session_state["skipped_col_filter"] = True

        if st.session_state['user_researchquestion'] or st.session_state['skipped_col_filter']:

            if st.session_state['user_researchquestion']: # Only bother running the LLM if there is a research question
                possible_columns = list(st.session_state.genes_info_df.columns)
                # set up prompt:
                prompt = PromptTemplate( # Can improve prompt and use o1 preview with it soon
                template= """
                - Here is a list of column names in a dataframe: {col_names}
                - The columns hold information relating to gene names, disease associations, biological processes, and much more.
                - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
                - Consult the following dictionary to understand the meanings of columns you initially do not understand: {colmeta_dict}\n
                - Here is the user's research question or hypothesis: {query}
                Instructions:
                - Using the query and the list of column names, select any column names you think might be relevant to their question or future analysis
                - Return only the column names relevant to the query in a list format. Remember, it is better to give more columns than necessary than to give not enough.
                To do this, it may actually be easier to think which to exclude because they are most likely irrelevant. For example, you may almost always
                want to include localization or expression columns because those can be very useful for answering future questions the user may have.
                Format instructions: Return ONLY a list in the format 'col1,col2,col3' (without the quotes, and with no brackets or anything)
                """
                )
        
                chain = prompt | llm_4o 
                parser_output = chain.invoke({"query": st.session_state['user_researchquestion'], "col_names": possible_columns,"colmeta_dict": st.session_state.genes_colmeta_dict})
                # st.write(parser_output)
                colnames_list = parser_output.content.split(",")
                # st.write(colnames_list)
                relevant_cols_only_df = st.session_state.genes_info_df[colnames_list]
                st.session_state['relevant_cols_only_df'] = relevant_cols_only_df

            st.session_state['user_refined_df'] = st.session_state['relevant_cols_only_df'] # At this stage since refinement hasn't been done yet, the refined_df should always be the relevant cols df

            # Display either column-refined df or the df with all columns
            st.subheader("Data with columns relevant to your research question:")    
            st.dataframe(st.session_state['relevant_cols_only_df'])

            st.divider()
            st.header("Refine your results",divider='green')
            refine_box = st.container(height=150)
            with refine_box:
                user_refinement_q = st.text_input("E.g. 'Only keep genes with low tissue specificity'",max_chars=500)
                if user_refinement_q:
                    st.session_state['user_refinement_q'] = user_refinement_q
                st.write("**Your Data Refinement Query:** ",st.session_state['user_refinement_q'])

            # Show option to skip if user has not yet entered anything
            if not st.session_state['user_refinement_q']:
                col1, col2 = st.columns([0.7,0.3])
                with col1:
                    if not st.session_state['skipped_initial_refine']:
                        st.markdown(st.session_state.provide_refinement_text)
                with col2:
                    if st.button("Keep all rows/genes", use_container_width=True, on_click=clear_text(text_element='ref_prompt')):
                        st.session_state['user_refined_df'] = st.session_state['relevant_cols_only_df'] # If the user skips the first refinement, the refined df is just the df with relevant columns
                        st.session_state["skipped_initial_refine"] = True
            elif st.session_state['user_refinement_q']:

                st.session_state['skipped_initial_refine'] = False

                pd_df_agent = create_pandas_dataframe_agent(
                    llm=llm_4o,
                    df=st.session_state['relevant_cols_only_df'],
                    agent_type="tool-calling", # Significantly faster and less likely to run into parsing errors than the default
                    verbose=True,
                    allow_dangerous_code=True, 
                    include_df_in_prompt=True, # could possibly remove to save tokens
                    number_of_head_rows=10
                )
                pd_df_agent.handle_parsing_errors = "Check your output and make sure it conforms, use the Action Input/Final Answer syntax"
                full_prompt = f"""
                User refining statement: {st.session_state['user_refinement_q']}
                Instructions: 
                - Given the user refinement statement above and the dataframe you were given, return the pandas expression required to achieve this.
                - Keep in mind some column values may be comma or otherwise delimited and contain multiple values.
                - Return only the code in your reply
                - Do not include any additional formatting, such as markdown code blocks or backticks (`).
                - For formatting, do not allow any lines of code to exceed 80 columns
                - Example: E.g. you might return: df_y = dfx[dfx['blah'] == 'foo']
                """
                # blah = pd_df_agent.aprep_inputs
                # print(blah)
                response = pd_df_agent.run(full_prompt)
                # response = pd_df_agent.run(st.session_state['user_refinement_q'])
                # st.write(response) # FOR DEBUGGING LLM OUTPUT
                pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
                relevant_cols_only_df = st.session_state['relevant_cols_only_df'] #added because original variable is now out of context
                pandas_code_only = pandas_code_only.replace("df", "relevant_cols_only_df")
                pandas_code_only = pandas_code_only.rstrip('`') # remove code backticks left over
                # st.write(f"Code to be evaluated:{pandas_code_only}") # FOR DEBUGGING LLM OUTPUT
                user_refined_df = eval(pandas_code_only)
                st.session_state['user_refined_df'] = user_refined_df
                

            if st.session_state['user_refinement_q'] or st.session_state['skipped_initial_refine']: # Only show buttons if user has either given a refinement or skipped that step

                # Show either the updated refined dataframe or the full one with whatever columns were filtered (if any)
                st.subheader("Data with any row/gene refinements:")    
                st.dataframe(st.session_state.user_refined_df)
                st.divider()
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
        if 'show_refine_chat_buttons' not in st.session_state:
            st.session_state['show_refine_chat_buttons'] = True
        if 'data_chat' not in st.session_state:
            st.session_state['data_chat'] = False
        if 'analyze_data' not in st.session_state:
            st.session_state['analyze_data'] = False

        ### REPEAT REFINEMENT: 
        
        if st.session_state['do_refine_loop']:
            repeat_refinement(llm=llm_4o)

        if st.session_state['data_chat']:
            chat_with_data(llm=llm_4o)

        if st.session_state['analyze_data']:
            analyze_data(llm=llm_4o)

        left_col, right_col = st.columns(2)
        chat_button_ph = st.empty()
        analyze_button_ph = st.empty()

        if st.session_state['show_chat_analyze_buttons']:
            left_col.button("Chat with your Data", icon="💬", use_container_width=True,on_click=chat_buttonclick)
            right_col.button("Ready To Analyze", icon="🔬", use_container_width=True,on_click=analyze_buttonclick)

        if st.session_state['show_refine_analyze_buttons']:
            left_col.button("Keep Refining", icon="🔃", use_container_width=True,on_click=refineloop_buttonclick)
            right_col.button("Ready To Analyze", icon="🔬", use_container_width=True,on_click=analyze_buttonclick)

        if st.session_state['show_refine_chat_buttons']:
            left_col.button("Keep Refining", icon="🔃", use_container_width=True,on_click=refineloop_buttonclick)
            right_col.button("Chat with your Data", icon="💬", use_container_width=True,on_click=chat_buttonclick)
        
