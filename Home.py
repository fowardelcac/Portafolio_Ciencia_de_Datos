import streamlit as st
from PIL import Image
from io import BytesIO
import requests

st.set_page_config(
    page_title = "Portafolio",
    page_icon = "🔥",
)



def main():
    st.title("Bienvenidos a mi portafolio.")
  

    st.markdown('''Soy Juan Cruz, estudiante de Licenciatura en Ciencia de Datos. Cuento con conocimientos de programación tanto en Python como R; especificamente en python trabajo con librerías como Pandas, Matplotlib/Seaborn, Numpy, Sklearn y Machine Learning. Busco oportunidades laborales en el campo de la ciencia de datos, con interés en el sector tecnológico. Soy autodidacta, cuento nivel intermedio de inglés y estoy ansioso por colaborar en proyectos desafiantes. ¡Gracias por tu consideración!''')    
    
    st.subheader('Descripcion:')
    st.markdown("Este es mi portafolio de Ciencia de Datos, el cual consta de algunos proyectos destacados:")
    st.markdown('''
                Contenidos
PBI dashboard: Este es un tablero de control creado en Python que presenta gráficos simples relacionados con el Producto Interno Bruto (PIB) a nivel mundial y regional. El objetivo principal de este proyecto es demostrar cómo se puede crear un tablero de control simple, claro y conciso utilizando Python y generar visualizaciones a través de Plotly. Su repositorio original: https://github.com/fowardelcac/PBI
Analisis de sentimiento: El objetivo principal de este proyecto fue realizar un análisis de sentimiento en relación a cinco prominentes políticos argentinos. Este análisis se basó en la evaluación de tweets en los cuales los usuarios los mencionaban. La primera fase del proyecto involucró la extracción de tweets, específicamente aquellos publicados los días 13 y 14 de abril. Utilicé la biblioteca Snscrape para llevar a cabo esta tarea de manera eficiente y precisa. En la segunda etapa, trabajé en el procesamiento de los datos recolectados. Utilicé un modelo de Procesamiento de Lenguaje Natural (NLP) conocido como roBERTa, el cual me permitió llevar a cabo un análisis de sentimiento. Este análisis generó tres posibles puntajes: 1) Negativo, 2) Neutro y 3) Positivo, proporcionando una visión completa de las opiniones expresadas en los tweets. Finalmente, con base en los resultados obtenidos, elaboré conclusiones que contribuyeron a una comprensión más profunda de la percepción pública en torno a estos políticos. Este proyecto no solo demostró mis habilidades analíticas y mi capacidad para trabajar con herramientas avanzadas de procesamiento de datos, sino también mi capacidad para llevar a cabo investigaciones significativas y generar información valiosa a partir de datos no estructurados. Su repositorio original: https://github.com/fowardelcac/Analisis-de-sentimiento-politicos-argentinos
Análisis de Direcciones de Ethereum: En este proyecto, me adentré en el emocionante mundo de las criptomonedas y me conecté a la API de Etherscan para crear una aplicación interactiva. Esta aplicación permite interactuar con la API para obtener datos a partir de las direcciones de Ethereum. Entre los datos disponibles se encuentran el saldo de Ethereum de cada dirección, el historial de transacciones y los tokens que poseen.vEl propósito principal de este proyecto es demostrar mi habilidad para acceder a datos utilizando una API y organizarlos de manera que proporcionen información valiosa y conocimientos significativos. A través de esta aplicación, no solo he obtenido datos detallados sobre las direcciones de Ethereum, sino que también he logrado visualizar insights importantes que pueden ser de utilidad tanto para investigadores como para entusiastas de las criptomonedas. Explorando las finanzas descentralizadas, la tecnología blockchain y el fascinante mundo de Ethereum, este proyecto es un testimonio de mi capacidad para trabajar con datos complejos y presentarlos de manera clara y efectiva para la toma de decisiones informadas. Repositorio original: https://github.com/fowardelcac/Analisis-ethereum-address
                ''')
    
    st.subheader("Como contactarme:")
    st.markdown('''
                1. Linkedin: https://www.linkedin.com/in/jcs2
                2. Github: https://github.com/fowardelcac
                3. Gmail: juansaldano01@gmail.com
                4. WhatsApp: 3515126823
                ''')
    
    
    image_url = 'https://raw.githubusercontent.com/fowardelcac/Portafolio_Ciencia_de_Datos/main/70445613.jpg'
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.sidebar.image(image, width=160)
     # Aplicar CSS personalizado para alinear la imagen a la izquierda en el sidebar
    st.sidebar.markdown(
        """
        <style>
        div.stSidebar > div.stElement > div.stImage > img {
            float: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

if __name__ == '__main__':
    main()
