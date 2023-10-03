import streamlit as st
from PIL import Image
from io import BytesIO
import requests

st.set_page_config(
    page_title="Portafolio",
    page_icon="🔥",
)

def main():
    st.title("¡Bienvenidos a mi Portafolio!")

    st.markdown('''
    Soy Juan Cruz, un apasionado estudiante de Licenciatura en Ciencia de Datos. Mi enfoque se centra en la programación, tanto en Python como en R, donde destaco en el manejo de librerías como Pandas, Matplotlib/Seaborn, Numpy, Sklearn y Machine Learning. Estoy en búsqueda de oportunidades laborales en el apasionante campo de la Ciencia de Datos, con especial interés en el sector tecnológico.
    
    Soy autodidacta y cuento con un nivel intermedio de inglés. Estoy ansioso por colaborar en proyectos desafiantes y seguir aprendiendo en este emocionante camino. ¡Gracias por tu consideración!
    ''')

    st.subheader('Descripción:')
    st.markdown("Mi Portafolio de Ciencia de Datos presenta una variedad de proyectos destacados que incluyen:")

    st.markdown('''
    - **PBI Dashboard**: Un tablero de control creado en Python que presenta gráficos simples relacionados con el Producto Interno Bruto (PIB) a nivel mundial y regional. El objetivo principal es demostrar cómo se puede crear un tablero de control simple y efectivo utilizando Python y generando visualizaciones con Plotly. [Ver repositorio](https://github.com/fowardelcac/PBI)
    
    - **Análisis de Sentimiento**: Este proyecto se centró en realizar un análisis de sentimiento en relación a cinco prominentes políticos argentinos. Se basó en la evaluación de tweets en los cuales los usuarios los mencionaban. El proyecto incluye la extracción de tweets, procesamiento de datos y el uso de un modelo de Procesamiento de Lenguaje Natural (NLP) llamado roBERTa para generar análisis de sentimiento. [Ver repositorio](https://github.com/fowardelcac/Analisis-de-sentimiento-politicos-argentinos)
    
    - **Análisis de Direcciones de Ethereum**: Exploré el emocionante mundo de las criptomonedas, conectándome a la API de Etherscan para crear una aplicación interactiva. Esta aplicación permite obtener datos a partir de las direcciones de Ethereum, incluyendo saldos, historiales de transacciones y tokens en posesión. [Ver repositorio](https://github.com/fowardelcac/Analisis-ethereum-address)
    ''')

    st.subheader("Cómo contactarme:")
    st.markdown('''
    - **LinkedIn**: [Perfil de LinkedIn](https://www.linkedin.com/in/jcs2)
    - **GitHub**: [Perfil de GitHub](https://github.com/fowardelcac)
    - **Correo Electrónico**: juansaldano01@gmail.com
    - **WhatsApp**: +54 351 512 6823
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
