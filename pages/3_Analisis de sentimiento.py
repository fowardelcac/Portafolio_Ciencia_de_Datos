import streamlit as st

st.title("Análisis de Sentimiento de Políticos Argentinos en Twitter")

st.markdown('''
El objetivo principal de este proyecto fue realizar un análisis de sentimiento en relación a cinco prominentes políticos argentinos. Este análisis se basó en la evaluación de tweets en los cuales los usuarios los mencionaban.

La primera fase del proyecto involucró la extracción de tweets, específicamente aquellos publicados los días 13 y 14 de abril. Utilicé la biblioteca Snscrape para llevar a cabo esta tarea de manera eficiente y precisa.

En la segunda etapa, trabajé en el procesamiento de los datos recolectados. Utilicé un modelo de Procesamiento de Lenguaje Natural (NLP) conocido como roBERTa, el cual me permitió llevar a cabo un análisis de sentimiento. Este análisis generó tres posibles puntajes: 1) Negativo, 2) Neutro y 3) Positivo, proporcionando una visión completa de las opiniones expresadas en los tweets.

Finalmente, con base en los resultados obtenidos, elaboré conclusiones que contribuyeron a una comprensión más profunda de la percepción pública en torno a estos políticos. Este proyecto no solo demostró mis habilidades analíticas y mi capacidad para trabajar con herramientas avanzadas de procesamiento de datos, sino también mi capacidad para llevar a cabo investigaciones significativas y generar información valiosa a partir de datos no estructurados.
''')

st.markdown('Link: [Leer más sobre el proyecto](https://medium.com/@juancruzsaldano9/imagen-de-los-pol%C3%ADticos-en-twitter-6f06ac3f956b)')

st.write('-'*100)
