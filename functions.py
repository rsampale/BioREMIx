import streamlit as st
import time
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import requests
from openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler # deprecated

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
            st.title("Welcome to BioREMIx")
        with help_placeholder:
            with st.expander("**⚠️ Read if You Need Help With Password**"):
                st.write("To request or get an updated password contact developers.")
            
                st.write("**Remi Sampaleanu** remi@wustl.edu")
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

                        

def reboot_hypothesizer():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
            
    # Iterate over the keys
    for key in keys:
        # If the key is not 'authenticated', delete it from the session_state
        if key not in ['authenticated','genes_info_df','genes_colmeta_dict','colmeta_dict','colmeta_df']:
            del st.session_state[key]
            
def refineloop_buttonclick(): # SHOULD MAKE THESE JUST TOGGLE MAYBE INSTEAD OF HARDCODING TRUE OR FALSE
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = True
    st.session_state['show_chat_analyze_buttons'] = True
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['show_refine_chat_buttons'] = False
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = False

def chat_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = True
    st.session_state['show_refine_chat_buttons'] = False
    st.session_state['data_chat'] = True
    st.session_state['analyze_data'] = False

def analyze_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['show_refine_chat_buttons'] = True
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = True

def clear_text(text_element):
    if text_element == 'rq_prompt':
        st.session_state['provide_rq_text'] = ""
    elif text_element == 'ref_prompt':
        st.session_state['provide_refinement_text'] = ""

def submit_text(location):
    if location == 'initial_refinement':
        st.session_state.user_refinement_q = st.session_state.init_refinement_q_widget
        st.session_state.last_refinement_q = st.session_state.init_refinement_q_widget
        st.session_state.init_refinement_q_widget = None
    elif location == 'repeat_refinement':
        if st.session_state.repeat_refinement_q_widget: # Only update the last refinement if the widget is not empty
            st.session_state.last_refinement_q = st.session_state.repeat_refinement_q_widget
        st.session_state.user_refinement_q = st.session_state.repeat_refinement_q_widget
        st.session_state.repeat_refinement_q_widget = None
        
def undo_last_refinement(refinement):
    # st.write("EXECUTED UNDO")
    if refinement == "initial":
        if len(st.session_state.gene_df_history) >= 1:
            st.session_state.gene_df_history.pop()
        st.session_state.user_refinement_q = None
        st.session_state.skipped_initial_refine = False
    elif refinement == "repeat":
        if len(st.session_state.gene_df_history) > 1:
            st.session_state.gene_df_history.pop()
            st.session_state.user_refined_df = st.session_state.gene_df_history[-1][0] # Gets the df part of the most recent tuple in the history
            st.session_state.last_refinement_q = st.session_state.gene_df_history[-1][1]

# builds a pie chart for disease association
def build_visual_1(llm):
    all_columns = list(st.session_state.merged_df.columns)
    prompt = PromptTemplate(
    template = """
    - Here is a list of columns in a dataframe: {all_cols}
    - The columns hold information relating to gene names, disease associations, biological processes, and much more.
    - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
    Instructions:
                - Using the list of column names, select any column names you think might be relevant to diseases associated with a gene
                - Try to find columns related to neurodegenerative diseases such as SMA, SCA, ALS, juvenile ALS, parkinson, alzeimer, HSP, CMT, dHMN, and possibly others if they are present
                - Do not include any super ambiguous column name like curated diseases, any disease designation, or other diseases. Only include column names that seem related to very specific diseases
                - Do not include cancer or diabetes related columns
                - Never include the column for a gene name. Only include columns for disease names
                - Return two lists in a tuple. The first should be the real column names, the second should be the plot labels for these names (thus should be better formatted without underscores, etc.)
                - E.g. Return: (['colname_1','colname_2'],['col_label1','col_label2'])
                - Return ONLY the tuple. Do not add the word python or any quotations. 
    """
    )
    chain = prompt | llm
    parser_output = chain.invoke({"all_cols": all_columns})
    
    # recasts parser_output_content as a tuple
    parser_output_content = ast.literal_eval(parser_output.content)
    
    # separates lists in the tuple
    colnames_list=parser_output_content[0]
    colnames_labels=parser_output_content[1]

    # creates new df with only disease columns and adds to session state
    relevant_cols_only_df = st.session_state.merged_df[colnames_list]
    st.session_state['relevant_cols_only_df'] = relevant_cols_only_df

    # counts disease associations
    string_counts = relevant_cols_only_df.apply(lambda col: (col == 1).sum())
    
    # makes sure there is data to display (prevents displaying labels for 0 slices)
    nonzero_indices = string_counts > 0
    filtered_counts = string_counts[nonzero_indices]
    filtered_labels = [label for label, keep in zip(colnames_labels, nonzero_indices) if keep]
    if len(filtered_counts) == 0:
        st.write("You do not have any data to plot. Try to redo your refinement.")
        return
    
    # configures pie chart
    colormap=cm.get_cmap("Greens",len(colnames_list))
    colors=[colormap(i / len(colnames_list)) for i in range(len(colnames_list))]
    plt.figure(figsize=(6, 6))
    plt.pie(filtered_counts, labels=filtered_labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 5, }, labeldistance=1.05, colors=colors, wedgeprops={"edgecolor": "black", "linewidth": 1})
    plt.title("Disease Associations")
    plt.axis('equal')  
    
    st.pyplot(plt)

#builds a bar chart
def build_visual_2(llm):
    all_columns = list(st.session_state.merged_df.columns)
    prompt = PromptTemplate(
    template = """
    - Here is a list of columns in a dataframe: {all_cols}
    - The columns hold information relating to gene names, disease associations, biological processes, and much more.
    - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
    Instructions:
                - Using the list of column names, select one column the you think is most relevant to subcellular location.
                - Return the of the columns you found for subcellular location exactly as they are titled in the dataframe
                - E.g. Return: ('colname_1')
                - Return ONLY the column name with that formatting. Do not add any special characters or quotations
    """
    )
    chain = prompt | llm
    parser_output = chain.invoke({"all_cols": all_columns})
    
    parser_output_content = parser_output.content.strip('"')

    # creates list of subcellular locations and gets counts for each location
    all_locations = st.session_state.merged_df[parser_output_content].dropna().str.split(",")
    flat_locations = [loc.strip() for sublist in all_locations for loc in sublist]
    location_counts = pd.Series(flat_locations).value_counts()

    # makes sure there is actually data to plot
    nonzero_indices = location_counts > 0
    filtered_counts = location_counts[nonzero_indices]
    if len(filtered_counts) == 0:
        st.write("You do not have any data to plot. Try to redo your refinement.")
        return
    
    #configure bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    location_counts.plot(kind="bar", color="palegreen", edgecolor="black", ax=ax)
    plt.xlabel("Subcellular Location", fontsize=12)
    plt.ylabel("Number of Genes", fontsize=12)
    plt.title("Distribution of Genes Across Subcellular Locations", fontsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)


def repeat_refinement(llm):
    if 'repeat_refinement_q_widget' not in st.session_state:
        st.session_state['repeat_refinement_q_widget'] = None

    st.subheader("Enter your refining statement:")
    repeat_refine_box = st.container(height=150)
    with repeat_refine_box:
        st.text_input("E.g. 'Only keep genes with low tissue specificity'",max_chars=501,key='repeat_refinement_q_widget',on_change=submit_text(location="repeat_refinement")) # if maxchars = 500 it thinks its the same text_input as before
        st.write("**Your Most Recent Data Refinement Query:** ",st.session_state.last_refinement_q)
    
    ## repeat-refining agent:
    if st.session_state['user_refinement_q']: # only bother re-creating the dataframe if the user has given new input
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
        full_prompt = f"""
                User refining statement: {st.session_state['user_refinement_q']}
                Instructions: 
                - Given the user refinement statement above and the dataframe you were given, return the pandas expression required to achieve this.
                - Keep in mind some column values may be comma or otherwise delimited and contain multiple values.
                - Return only the code in your reply
                - Do not include any additional formatting, such as markdown code blocks
                - For formatting do not allow any lines of code to exceed 80 columns
                - Example: E.g. you might return df_y = dfx[dfx['blah'] == 'foo']
                """
        response = pd_df_agent.run(full_prompt)
        # st.write(response)
        pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
        pandas_code_only = pandas_code_only.replace("df", "st.session_state['user_refined_df']")
        pandas_code_only.replace("```", "").strip() # remove code backticks left over
        # st.write(f"Code to be evaluated:{pandas_code_only}")
        user_refined_df = eval(pandas_code_only)
        st.session_state['user_refined_df'] = user_refined_df

        # Add to history
        st.session_state.gene_df_history.append((st.session_state['user_refined_df'],st.session_state['last_refinement_q']))
        st.session_state['last_refinement_q'] = st.session_state.gene_df_history[-1][1]
    
    st.subheader("Instructions:")
    st.write("**Press enter to submit a refinement. Repeat as many times as needed.**")
    st.header("Current refined data:")
    # st.write(f"len of gene_df_history: {len(st.session_state.gene_df_history)}")
    # st.write(f"Query at top of history: {st.session_state.gene_df_history[-1][1]}")
    st.dataframe(st.session_state.user_refined_df) # maybe change to point to most recent df on history?
    st.button("**Undo** the last refinement",use_container_width=True,icon=":material/undo:",type="primary",on_click=undo_last_refinement,args=("repeat",))

    st.divider()

def chat_with_data(llm):
    
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key is not set. Please set the PERPLEXITY_API_KEY environment variable.")
    perplex_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    begeneral_prefix = """If the user is asking a specific question that can be answered from the df given to you, do so. Keep in mind some column values may actually be
    delimited lists or contain multiple values. If the user seems to be asking 
    a more general question about a gene, possible associations, biological process, etc. just use your internal knowledge. You can also mix the two sources of info,
    but then be clear where you are getting your information from. Try to keep responses relatively short unless asked for more information or told otherwise.

    Do not simulate data. You are an agent that has access to the real dataframe and can simply access it as the variable df.
    NEVER INCLUDE BACKQUOTES ('`') IN YOUR OUTPUT.
    """
    # Note that you have two dataframes you have access to. One contains genes and annotated info about those genes, and the other contains a user-uploaded gene expression
    # table with genes, logFC, padj, disease, and cell_type. Only use this expression dataframe if the user asks a question that requires it for an answer. When using it, consider only the genes
    # also present in the first dataframe.

    online_search = st.sidebar.toggle("Toggle Perplexity Online Search")

    non_toolcalling_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
        prefix=begeneral_prefix,
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=15
    )
    non_toolcalling_agent.handle_parsing_errors = "Check your output and make sure it answers the user query, use the Action Input/Final Answer syntax"

    pd_df_agent = create_pandas_dataframe_agent( # 'SIMULATES' the data instead of really using the df unless made very clear it has access to df in the prefix
        llm=llm,
        df=st.session_state['user_refined_df'],
        prefix=begeneral_prefix,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        # include_df_in_prompt=True,
        # number_of_head_rows=5
    )
    # pd_df_agent.handle_parsing_errors = True

    if "messages" not in st.session_state or st.sidebar.button("Clear chat history",use_container_width=True):
        st.session_state["messages"] = [{"role": "system", "content": "You are an AI assistant tasked to help a user gain insights from their data, and answer any of their questions. You already have access to a dataframe full of genes and annotation data to help."}]
    
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])
            # writes the user's progress -S

    if prompt := st.chat_input(placeholder="What is this data about?"):
        # Tack on instructions to the beginning of prompt HERE
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        
        # with st.expander("session_state.messages:",expanded=False):
        #         st.write(st.session_state.messages)
        # st.write(len(st.session_state.messages)) # is 1 before user provides anything

        if len(st.session_state.messages) > 1:
            # USE INTERNET/PERPLEXITY IF TOGGLE IS ON
            if not online_search:
                if st.session_state.messages[-1]["role"] == "user": 
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False,max_thought_containers=5)
                    # try:
                    #     response = non_toolcalling_agent.run(st.session_state.messages, callbacks=[st_cb]) # Has more error loops with certain queries
                    # except:
                    response = pd_df_agent.run(st.session_state.messages, callbacks=[st_cb]) # Still can't access the internet to provide specifics on studies etc.
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
            else:
                # use perplexity for the response instead
                if st.session_state.messages[-1]["role"] == "user": # Needs user-system alternating, only get response if last message was a user one
                    response = perplex_client.chat.completions.create(model="sonar",messages=st.session_state.messages)
                    response_content = response.choices[0].message.content
                    # Add on the links when actually displaying the response:
                    response_links = response.citations # A list of strings (links)
                    numbered_links = "\n".join(f"{i+1}. {link}" for i, link in enumerate(response_links))
                    final_response = f"{response_content}\n\n{numbered_links}"
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.write(final_response)

    # Put expander with the data at the bottom:
    with st.expander("**Click to view data being referenced**"):
        st.dataframe(st.session_state['user_refined_df'])

    st.divider()

def send_genesdata():
    gene_list = list(st.session_state['user_refined_df']['Gene_Name']) # HARDCODED SO WONT WORK IF USER DF DOESNT HAVE THIS NAME FOR GENE COLUMN
    
    # DDB_GENESLIST_API_URL = "https://9icpbd78nf.execute-api.us-east-1.amazonaws.com/prod"
    DDB_GENESLIST_API_URL = "https://9icpbd78nf.execute-api.us-east-1.amazonaws.com/proxy_prod"
    
    payload = {
        "values": gene_list  # Only include the values list here
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(f"{DDB_GENESLIST_API_URL}/store_data", json=payload, headers=headers)
    response_json = response.json()
    # st.write(response_json)
    # st.write(response)
    if response.status_code == 200:
        session_id = response_json["session_id"]
        shiny_url = f"http://98.84.134.224/NeuroKinex/?session_id={session_id}"
        st.session_state["neurokinex_url"] = shiny_url
        # st.markdown(f"[Go to Shiny App]({shiny_url})")
        st.markdown(f"WORKED: {session_id}")
    else:
        # st.write(response_json)
        st.error("Failed to store data.")

def analyze_data(llm):
    if 'merged_df' not in st.session_state:
         st.session_state['merged_df'] = None

    #create a merged data frame and add to session state
    common_columns = list(set(st.session_state.user_refined_df.columns) & set(st.session_state.genes_info_df.columns))
    defined_merged_df = st.session_state.user_refined_df.merge(st.session_state.genes_info_df, on=common_columns, how="inner")
    defined_merged_df = defined_merged_df.loc[:, ~defined_merged_df.columns.duplicated()]

    st.session_state['merged_df'] = defined_merged_df

    st.divider()
    st.title("Data Visualization")
    st.subheader("Here are some suggested visualizations that might be of use to you:")

    col1, col2 = st.columns(2)

    if 'chart_type' not in st.session_state:
        st.session_state.chart_type = None

    with col1:
        if st.button("Pie Chart: Disease Associations"):
            st.session_state.chart_type = "pie"

    with col2:
        if st.button("Bar Chart: Subcellular Location"):
            st.session_state.chart_type = "bar"

    if st.session_state.chart_type == "pie":
        build_visual_1(llm=llm)
    if st.session_state.chart_type == "bar":
        build_visual_2(llm=llm)
    with st.expander("**Click to view data being referenced**"):
        st.dataframe(st.session_state['merged_df'])

    st.divider()
    
    send_genes_placeholder = st.empty()
    with send_genes_placeholder:
        if st.button("Send your genes to NeuroKinex",use_container_width=True):
            send_genesdata()
            st.link_button(label="View Genes in NeuroKinex",url=st.session_state["neurokinex_url"],type="primary",use_container_width=True)

    st.divider()