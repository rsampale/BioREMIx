import streamlit as st

st.set_page_config(page_title='BioREMIx', layout = 'centered', page_icon = ':dna:', initial_sidebar_state = 'auto')
mgi_icon = "images/the_elizabeth_h_and_james_s_mcdonnell_genome_institute_logo.jpg"
st.logo(mgi_icon, size='large')

st.title("Welcome to BioRemix")
st.divider()
st.subheader("What is BioRemix?")
st.markdown("""
            BioRemix is an interactive data exploration platform! By leveraging Large Language Models (LLMs) and AI, we are able very effectively query genomic data and help our users formulate new hypotheses!
            
            """)
st.subheader("How does it work?")
st.markdown("""
            There are three main steps and functionalities currently implemented in BioRemix:

            1. Refine your data. Work with an LLM to iteratively narrow down your data to only include rows (genes) or columns that are of interest to you.
            2. Chat with your data. Freely question and speak with your data to find interesting trends, patterns, or genes.
            3. Analyze your data. Get AI generated graphs and visualizations suited to your research objectives, or view your genes of interest in our other tools (e.g. NeuroKinex).
            
            """)

st.divider()
with st.expander("**Latest Updates and Changes (09/05/25):**"):
    st.markdown("""
                - Added a GSEA tool in the analysis page, allowing users to perform Gene Set Enrichment Analysis on their filtered genes or a custom gene list.
                - Added a Tissue Coexpression Visualization tool, which lets users see which genes are most highly co-expressed with their gene of interest across different tissues.
                - Fixed bugs related to handling user-uploaded data.
                - Improved the UI and added additional helpful tooltips throughout the application.
                """)
with st.expander("**▶ Watch Tutorial Video**",expanded=True):
    st.video("https://youtu.be/MUc_28f-k5U")