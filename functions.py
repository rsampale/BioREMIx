import streamlit as st
import time
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
        # st.write(response)
        pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
        pandas_code_only = pandas_code_only.replace("df", "st.session_state['user_refined_df']")
        pandas_code_only = pandas_code_only.rstrip('`') # remove code backticks left over
        # st.write(f"Code to be evaluated:{pandas_code_only}")
        user_refined_df = eval(pandas_code_only)
        st.session_state['user_refined_df'] = user_refined_df
    
    st.subheader("Instructions:")
    st.write("**Press enter to submit a refinement. Repeat as many times as needed.**")
    st.header("Current refined data:")
    st.dataframe(st.session_state['user_refined_df'])

def chat_with_data(llm):
    pd_df_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
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
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pd_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

def analyze_data(llm):
    st.subheader("Restate your research objectives, if desired:")
    st.divider()
    st.subheader("Here are some suggested visualizations that might be of use to you:")