import streamlit as st

st.title("Otros proyectos:")
st.subheader('Analisis de sentimiento de politicos argentinto en twitter: ')
st.markdown('''
            El objetivo de este proyecto es realizar un analisis de sentimiento de cinco politicos argentinos, partiendo de tweets donde sus usuarios se encuentran mencionados. La primera parte del proyecto, se realiza la extraccion de tweets del dia 13 y 14 de Abril, utilizando la libreria Snscrape; en la segunda parte se procesas los datos y atravez de un modelo de NLP llamado roBERTa se obtiene un analisis de sentimento, el cual tiene tres posibles puntajes: 1) Negativo, 2) Neutro y 3) Positivo. Por ultimo se realiza una conslusion
            ''')
st.markdown('Link: https://medium.com/@juancruzsaldano9/imagen-de-los-pol%C3%ADticos-en-twitter-6f06ac3f956b')
st.write('-'*100)

st.subheader('ML con SKlearn:')
st.markdown('Implementacion de algoritmos de apredizaje supervizado/no superv., division de datasets e hyperparametrizacion utilizando la liberia Scikit learn')
st.markdown('Link: https://github.com/fowardelcac/Machine-learning-con-sklearn')
st.write('-'*100)