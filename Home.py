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
  

    st.markdown('''Soy Juan Cruz, estudiante de Licenciatura en Ciencia de Datos. Me apasiona la programación en Python y tengo experiencia en el desarrollo de aplicaciones web interactivas con Streamlit. Trabajo con librerías como Pandas, Matplotlib/Seaborn, Numpy, Sklearn y Yfinance para análisis de datos y Machine Learning. Busco oportunidades laborales en el campo de la ciencia de datos, con interés en el sector financiero y tecnológico. Soy autodidacta, tengo nivel intermedio de inglés y estoy ansioso por colaborar en proyectos desafiantes. ¡Gracias por tu consideración!''')    
    
    st.subheader('Descripcion:')
    st.markdown("Este es mi portafolio de Ciencia de Datos, el cual consta de tres proyectos destacados:")
    st.markdown('''
                1. Analizador de Ethereum Addresses: Este proyecto permite ingresar una dirección de Ethereum y obtener información relevante sobre sus transacciones, tokens ERC20 y NFTs asociados. Proporciona una visión detallada de la actividad y los activos digitales relacionados con una dirección específica en la blockchain de Ethereum.
                2. Optimizacion de portafolio: En este proyecto, se enfoca en la optimización de portafolios utilizando la frontera eficiente de Markowitz y el índice de Sharpe. Se analizan tres activos: MSFT, AAPL y TSLA, para obtener los pesos y valores óptimos del portafolio. El objetivo es maximizar el rendimiento y reducir el riesgo de la cartera de inversiones.
                3. First payment default: Este proyecto se centra en el análisis de riesgo crediticio utilizando un dataset sintético y un modelo de machine learning llamado LGBM. Los usuarios pueden ingresar sus datos en la página y el modelo determinará si son aptos para obtener una tarjeta de crédito o no.
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
