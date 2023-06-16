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
    st.markdown('''Este es mi portafolio de Ciencia de Datos, el cual consta de tres proyectos destacados.
    El primero es un análisis sobre Ethereum addresses, donde se puede ingresar una dirección y obtener datos relevantes sobre sus transacciones, tokens ERC20 y NFTs asociados. Este proyecto proporciona una visión detallada de la actividad y los activos digitales relacionados con una dirección específica en la blockchain de Ethereum.
    El segundo proyecto se enfoca en la optimización de portafolios, basandose en la frontera efieciente de markowitz y la optimización basada en el índice de Sharpe y la volatilidad, este programa permite ingresar la cantidad de activos y sus respectivos tickers para obtener los pesos y valores óptimos del portafolio. Con esto, se busca maximizar el rendimiento y reducir el riesgo de la cartera de inversiones.
    Por último, el tercer proyecto se centra en el análisis de riesgo crediticio.''')
    
    st.subheader("Como contactarme:")
    st.markdown('''
                1. Linkedin: https://www.linkedin.com/in/jcs2
                2. Github: https://github.com/fowardelcac
                3. Gmail: juansaldano01@gmail.com
                4. WhatsApp: 3515126823
                ''')
    
    
    image_url = 'https://raw.githubusercontent.com/fowardelcac/Portafolio-Data-Science/main/70445613.jpg'
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