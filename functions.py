import streamlit as st
import time
import ast
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler

def authenticate():
    # placeholders variables for UI 
    title_placeholder = st.empty()
    help_placeholder = st.empty()
    password_input_placeholder = st.empty()
    button_placeholder = st.empty()
    success_placeholder = st.empty()
    
    # check if not authenticated 
    if not st.session_state['authenticated']:
        # UI for authentication
        with title_placeholder:
            st.title("👋 Welcome to BioREMIx")
        with help_placeholder:
            with st.expander("**⚠️ Read if You Need Help With Password**"):
                st.write("To request or get an updated password contact developers.")
            
                st.write("""
**Remi Sampaleanu**
             
             remi@wustl.edu""")
            # UI and get get user password
            with password_input_placeholder:
                user_password = st.text_input("Enter the application password:", type="password", key="pwd_input")
            check_password = True if user_password == st.secrets["PASSWORD"] else False
            # Check user password and correct password
            with button_placeholder:
                if st.button("Authenticate") or user_password:
                    # If password is correct
                    if check_password:
                        st.session_state['authenticated'] = True
                        password_input_placeholder.empty()
                        button_placeholder.empty()
                        success_placeholder.success("Authentication Successful!")
                        st.balloons()
                        time.sleep(1)
                        success_placeholder.empty()
                        title_placeholder.empty()
                        help_placeholder.empty()
                    else:
                        st.error("❌ Incorrect Password. Please Try Agian.")
                        

def clear_session_state_except_password():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
            
    # Iterate over the keys
    for key in keys:
        # If the key is not 'authenticated', delete it from the session_state
        if key != 'authenticated':
            del st.session_state[key]
            
def refineloop_buttonclick(): # SHOULD MAKE THESE JUST TOGGLE MAYBE INSTEAD OF HARDCODING TRUE OR FALSE
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = True
    st.session_state['show_chat_analyze_buttons'] = True
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = False

def chat_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = True
    st.session_state['data_chat'] = True
    st.session_state['analyze_data'] = False

def analyze_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = True

def toggle_vizdict_build():
    st.session_state['viz_dict_create'] = False

def build_graph(name, desc, llm):
    prompt = f"""
    Here is a chart or visualization: {name}.
    Here is a brief and general description of what the chart could look like or compare: {desc}.
    Format instructions: Return ONLY the matplotlib/python code required to create this graph from the dataframe you are given.
    Do not create a new dataframe, and instead access the df you are given for data.
    Do not include ANYTHING that is not code (subtitles, descriptions, comments, etc.). 
    Do not include any formatting characters (i.e. backticks) in your response. Just plain text.
    """
    df_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
        agent_type="tool-calling", 
        verbose=True,
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=10
    )
    df_agent.handle_parsing_errors = "Check your output and make sure it follows the format instructions precisely. Use the Action Input/Final Answer syntax"

    response = df_agent.run(prompt)

    return response
    

def repeat_refinement(llm):
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = None

    st.subheader("Enter your refining statement:")
        
    repeat_refine_box = st.container(height=150)
    with repeat_refine_box:
        user_rerefinement_q = st.text_input("E.g. 'Only keep genes with low tissue specificity'",max_chars=501,value=st.session_state['input_text'],key='input_text') # if maxchars = 500 it thinks its the same text_input as before
        if user_rerefinement_q:
            st.session_state['user_refinement_q'] = user_rerefinement_q
        st.write("**Your Most Recent Data Refinement Query:** ",st.session_state['user_refinement_q'])
    
    ## repeat-refining agent:
    if st.session_state['input_text']: # only bother re-creating the dataframe if the user has given new input
        pd_df_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state['user_refined_df'],
            agent_type="tool-calling", # can also be others like 'openai-tools' or 'openai-functions'
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
        # st.write(response)
        pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
        pandas_code_only = pandas_code_only.replace("df", "st.session_state['user_refined_df']")
        pandas_code_only.replace("```", "").strip() # remove code backticks left over
        # st.write(f"Code to be evaluated:{pandas_code_only}")
        user_refined_df = eval(pandas_code_only)
        st.session_state['user_refined_df'] = user_refined_df
    
    st.subheader("Instructions:")
    st.write("**Press enter to submit a refinement. Repeat as many times as needed.**")
    st.header("Current refined data:")
    st.dataframe(st.session_state['user_refined_df'])

def chat_with_data(llm):

    prefix = f"""
    If you do not understand the meaning of a column by its name, consult the following 
    dictionary for column names and their descriptions: {st.session_state.colmeta_dict}
    """

    non_toolcalling_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=20
    )

    pd_df_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
        # prefix=prefix, # Putting the prefix here makes each query too long, because it tacks it on every time - find a way to load it into memory
        agent_type="tool-calling",
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=20
    )
    pd_df_agent.handle_parsing_errors = True

    if "messages" not in st.session_state or st.sidebar.button("Clear chat history",use_container_width=True):
        st.session_state["messages"] = [{"role": "assistant", "content": "What are you interested in discovering about the data?"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # st.write(len(st.session_state.messages)) # is 1 before user provides anything
        if len(st.session_state.messages) > 1:
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False,max_thought_containers=2)
            try:
                response = non_toolcalling_agent.run(st.session_state.messages, callbacks=[st_cb]) 
            except:
                response = pd_df_agent.run(st.session_state.messages, callbacks=[st_cb]) # Still can't access the internet to provide specifics on studies etc.
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

def analyze_data(llm):
    st.subheader("Restate your research objectives, if desired:")
    st.write("Default = the initial research question you provided")
    new_researchquestion = st.text_input("E.g. 'I am interested in finding where common ALS and PD genes localize'",max_chars=500)
    if new_researchquestion:
        st.session_state['user_researchquestion'] = new_researchquestion
    st.write(f"**Current research question:** {st.session_state.user_researchquestion}")

    st.divider()
    st.subheader("Here are some suggested visualizations that might be of use to you:")

    if 'viz_dict_create' not in st.session_state:
        st.session_state['viz_dict_create'] = True

    # Only build the dictionary of button names once
    if st.session_state.viz_dict_create:
        prompt = f"""
        You will be given a dataframe 'df' and user research question. Your goal is to suggest visualizations, graphs, charts, etc. to guide their research question and objectives.
        User research objective: {st.session_state.user_researchquestion}\n
        Output format instructions: A DICTIONARY of the graph type/name as the key, and a brief and non-specific description of the graph/visualization as the value. Do not reference specific columns.
        Include a maximum of 5 different visualizations.
        The graphs or charts must be able to be constructed from existing df columns.
        Always include a final key-value pair that is 'I have my own idea':'User will give their own graph suggestion'
        Output should not have any formatting characters.\n
        """
        viz_pandas_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state['user_refined_df'],
            agent_type="tool-calling",
            allow_dangerous_code=True,
            include_df_in_prompt=True,
            number_of_head_rows=10
        )
        viz_pandas_agent.handle_parsing_errors = "Check your output and make sure it conforms to the format instructions given, use the Action Input/Final Answer syntax."
        viz_dict_response = viz_pandas_agent.run(prompt)
        viz_dict_response.replace("```", "").strip() 
        # st.write(viz_dict_response)
        viz_dict_response = ast.literal_eval(viz_dict_response) # converts llm response (string) into the dictionary literal

        # Store dictionary to session state to avoid rebuilding it every time the page re-runs:
        if 'viz_dict' not in st.session_state:
            st.session_state['viz_dict'] = viz_dict_response

    st.write(st.session_state['viz_dict'])
    chart_names = list(st.session_state['viz_dict'].keys())
    chart_descriptions = list(st.session_state['viz_dict'].values())

    cols = st.columns(len(st.session_state['viz_dict'])-1) 
    for i, col in enumerate(cols):
        with col:
            if st.button(chart_names[i],use_container_width=True,on_click=toggle_vizdict_build):
                llm_graph_output = build_graph(chart_names[i],chart_descriptions[i],llm)
                reformatted_graph_code = llm_graph_output.replace("df", "st.session_state['user_refined_df']")
                reformatted_graph_code.replace("```", "").strip() # remove code backticks left over
                st.write(reformatted_graph_code)
                exec(reformatted_graph_code, globals())
                st.pyplot(plt)
    # Always have a 6th button that lets user suggest their own visualization
    make_own_placeholder = st.empty()
    with make_own_placeholder:
        if st.button(chart_names[-1],use_container_width=True,on_click=toggle_vizdict_build):
            st.write("pog")