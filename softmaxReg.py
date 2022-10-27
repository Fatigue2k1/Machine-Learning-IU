import streamlit as st

import datetime as dt

original_list = ['A','B', 'C']

result = st.selectbox('Search', original_list)

