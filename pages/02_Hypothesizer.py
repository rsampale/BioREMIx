import streamlit as st
from default_data import load_default_data
from functions import *
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from collections import deque 

# PAGE CONFIG
st.set_page_config(page_title="Gene Exploration Tool")

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
    if st.session_state.get('genes_info_df') is None:
        load_default_data()
    
    # Get API keys
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key is not set. Please set the PERPLEXITY_API_KEY environment variable.")
 
    st.sidebar.button("Reboot Session", on_click=reboot_hypothesizer,use_container_width=True,help="Completely reset the application and any refinements made.")


    if 'refine_section_visible' not in st.session_state:
        st.session_state['refine_section_visible'] = True
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
    if 'used_uploaded_goi' not in st.session_state:
        st.session_state["used_uploaded_goi"] = False
    if 'last_pandas_code' not in st.session_state: # So that users can see the most recent pandas code executed
        st.session_state['last_pandas_code'] = None
    if 'gene_df_history' not in st.session_state: # STORES PAST AND PRESENT GENE DFs TO ALLOW FOR UNDOs
        st.session_state['gene_df_history'] = deque(maxlen=7)

    # Text elements
    if 'provide_refinement_text' not in st.session_state:
        st.session_state['provide_refinement_text'] = "Please provide a refining statement to proceed."

    # CREATE MODELS
    llm_4o = ChatOpenAI(temperature=0, model='gpt-4o-2024-11-20', openai_api_key=OPENAI_API_KEY)
    llm_41 = ChatOpenAI(temperature=0, model='gpt-4.1', openai_api_key=OPENAI_API_KEY)
    llm_o4_mini = ChatOpenAI(temperature=0, model='o4-mini', openai_api_key=OPENAI_API_KEY) # no access yet as of 02/03/25
    rag_llm = ChatOpenAI(openai_api_key = st.secrets.OPENAI_API_KEY, model = "o4-mini")

    # PAGE FORMAT CODE START
    # Make into invisible container so it can be hidden with appropriate buttons?
    if 'do_refine_loop' not in st.session_state:
        st.session_state['do_refine_loop'] = None
        
    if st.session_state['refine_section_visible']:
        
        # Store the refinement query in a variable holding the last refinement_q so that it can be displayed in repeat_refine. This variable is NOT wiped during widget on_change
        if 'last_refinement_q' not in st.session_state:
            st.session_state['last_refinement_q'] = None
        
        st.title("Gene-Data Exploration Tool")
        
        initial_df = st.session_state['genes_info_df']
        st.session_state['relevant_cols_only_df'] = initial_df
 

        st.divider()
        
        st.subheader("How Does This Work?")
        st.markdown("""
                    The Hypothesiszer is BioRemix's bread and butter. It allows researchers to filter and query gene-annotation data and subset it to their
                    genes of interest. Users can then chat with their genetic data or send it to some of our other tools for more robust analyses.

                    1. (Optional) Upload a list of your Genes of Interest (GOI), or processed Differentially Expressed Genes (DEGs) in the **'Data Upload'** section.
                    2. Make your first refinement (filter). Note that you can subset both rows and/or columns.
                    3. Make a selection from the following: 'Keep Refining', 'Chat with your Data', or 'Ready to Analyze'
                        - If you would like to further narrow down your data, select 'Keep Refining'.
                        - If you would like to chat with your data or ask questions of it using human language, select 'Chat with your Data'.
                            - Note that if you have uploaded relevant expression data (logFC, pval, etc.), you may involve it in your queries here. 
                        - In the 'Ready to Analyze' section, you can find quick visualizations of interesting columns, and buttons to send your data/genes
                            to the BioMiners Tool Suite for more robust analyses and visualizations. 
                    4. Enjoy exploring your genetic data!
                    """)
        with st.expander("**Initial Data**",expanded=False):
            st.dataframe(st.session_state.genes_info_df)
            
        st.header("Refine your data",divider='green')
        st.info("Want to filter the data to a pre-set list of genes? You can do so in the **'Data Upload'** section!")
        # Check to see if the user has uploaded genes of interest
        if st.session_state.get('uploaded_goi_list') is not None:
            st.info("**We have detected that you have uploaded a list of genes of interest.**\n\nTo use the uploaded genes as your initial filter, click the button below. Otherwise, provide a refinement statement in the box below as usual.")
            st.button("Use uploaded genes as initial filter",use_container_width=True,help="Click to use the uploaded list of genes as your initial filter. You may still refine further after this step.",on_click=apply_uploaded_goi)
                
        # st.write(st.session_state['user_refinement_q'],st.session_state['used_uploaded_goi'],st.session_state['skipped_initial_refine']) # For testing

        refine_box = st.container(height=150)
        with refine_box:
            if st.session_state.get('init_refinement_q_widget') is None:
                if st.session_state['used_uploaded_goi']:
                    st.session_state.init_refinement_q_widget = st.session_state['user_refinement_q'] # Set the widget to initially render the fake refinement query made by using uploaded GOI, otherwise it is None and will turn user_refinement_q to None
                else:
                    st.session_state.init_refinement_q_widget = None
            st.text_input("E.g. 'Only keep genes involved in ALS'",max_chars=500,key='init_refinement_q_widget',on_change=submit_text(location='initial_refinement'),
                          help="Submit your first data refinement. Note that you may only submit one in this stage (see 'Keep Refining' to add more). It is generally recommended to submit refinements/filters one at a time. Complex refinements may occasionally fail.")
            st.write("**Your Data Refinement Query:** ",st.session_state['user_refinement_q'])

        # Show option to skip if user has not yet entered anything
        if not st.session_state['user_refinement_q']:
            col1, col2 = st.columns([0.7,0.3])
            with col1:
                if not st.session_state['skipped_initial_refine']:
                    st.markdown(st.session_state.provide_refinement_text)
            if st.session_state.used_uploaded_goi is False:
                with col2:
                    if st.button("Keep all rows/genes", use_container_width=True, on_click=clear_text(text_element='ref_prompt')):
                        st.session_state['user_refined_df'] = st.session_state['relevant_cols_only_df'] # If the user skips the first refinement, the refined df is just the df with relevant columns
                        st.session_state["skipped_initial_refine"] = True
        elif st.session_state['user_refinement_q'] and not st.session_state['used_uploaded_goi']:

            st.session_state['last_refinement_q'] = st.session_state.user_refinement_q

            st.session_state['skipped_initial_refine'] = False

            pd_df_agent = create_pandas_dataframe_agent(
                llm=llm_41,
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
            st.session_state['last_pandas_code'] = pandas_code_only
            st.session_state['user_refined_df'] = user_refined_df

        if st.session_state['user_refinement_q'] or st.session_state['skipped_initial_refine']: # Only show buttons if user has either given a refinement or skipped that step
            # Add the current user_refined_df (whether with rows filtered or not) to the history (used for repeat refine, not here):
            st.session_state.gene_df_history.append((st.session_state['user_refined_df'],st.session_state['user_refinement_q'],st.session_state['last_pandas_code'])) # Append to history a tuple of (df, most recent refinement, most recent code expr)

            # Show either the updated refined dataframe or the full one with whatever columns were filtered (if any)
            st.subheader("Data with any row/gene refinements:")
            # st.write(len(st.session_state.gene_df_history))
            with st.expander("**Click** to see most recent filter code",expanded=False):
                st.write(st.session_state['last_pandas_code'])
            st.dataframe(st.session_state.user_refined_df)
            
            # Button to undo last refinement:
            st.button("**Undo** the last refinement",use_container_width=True,icon=":material/undo:",type="primary",on_click=undo_last_refinement,args=("initial",))
            # st.write(f"len of gene_df_history: {len(st.session_state.gene_df_history)}")

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
            repeat_refinement(llm=llm_41)

        if st.session_state['data_chat']:
            chat_with_data(llm=llm_41, rag_llm=llm_41)

        if st.session_state['analyze_data']:
            analyze_data(llm=llm_41)

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
        
