import streamlit as st

st.set_page_config(page_title='BioREMIx', layout = 'centered', page_icon = ':dna:', initial_sidebar_state = 'auto')
mgi_icon = "images/the_elizabeth_h_and_james_s_mcdonnell_genome_institute_logo.jpg"
st.logo(mgi_icon, size='large')

st.title("Welcome to BioREMIx")
st.divider()
st.subheader("What is BioREMIx?")
st.markdown("""
            BioREMIx is an interactive data exploration platform. By leveraging Large Language Models (LLMs) and AI, we are able very effectively query genomic data and help our users formulate new hypotheses!
            
            """)
st.subheader("How does it work?")
st.markdown("""
            There are four main steps and functionalities currently implemented in BioREMIx:

            1. Narrow down your search space and data (your csv), to only include columns relevant to your objectives. This significantly speeds up the following steps.
            2. Refine your data. Work with an LLM to iteratively narrow down your data to only include rows (genes) or columns that are of interest to you.
            3. (Optional) Chat with your data. Freely question and speak with your data to find interesting trends, patterns, or genes.
            4. Analyze your data. Get AI recommended graphs and visualizations suited to your research objectives.
            
            """)