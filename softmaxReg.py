import streamlit as st

import datetime as dt

display_text = 'whatever i am typing here'
now = dt.datetime.now().strftime('$y-%m-$d %H:%M')

st.write(f'It is now {now} and I am typing: {display_text}')

whatever = 'your name'
print (f'This is {whatever} python file.')

original_list = ['Select Team','Parma', 'AC Milan']

result = st.selectbox('Select your favourite team', original_list)
st.write(f'La tua squadra favorita: {result}') 
