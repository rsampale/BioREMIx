import streamlit as st

st.set_page_config(page_title='BioREMIx', layout = 'centered', page_icon = ':dna:', initial_sidebar_state = 'auto')
mgi_icon = "images/the_elizabeth_h_and_james_s_mcdonnell_genome_institute_logo.jpg"
st.logo(mgi_icon, size='large')

st.title("Welcome to BioREMIx")
st.divider()
st.subheader("What is BioREMIx?")
st.subheader("How does it work?")