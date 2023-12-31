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
    Soy Juan Cruz, un estudiante de Licenciatura en Ciencia de Datos en la Universidad Siglo 21 de Cordoba, Argentina. Mi enfoque se centra en la programación con Pytho y cuento con 3 años de experiencia, donde destaco en el manejo de librerías como Pandas, Matplotlib/Seaborn, Numpy, y implementaciones de modelos de Machine Learning con Sklearn;  ademas, cuento con conocimientos basicos en R. Estoy en búsqueda de oportunidades laborales en el apasionante campo de la Ciencia de Datos, con especial interés en el sector tecnológico.
    
    Soy autodidacta y cuento con un nivel intermedio de inglés, me entusiasma colaborar en proyectos desafiantes que me permitan continuar desarollando mis habilidades y seguir mejorando en el tiempo. ¡Gracias por tu consideración!
    ''')

    st.subheader('Descripción:')
    st.markdown("Mi Portafolio de Ciencia de Datos presenta una variedad de proyectos destacados que incluyen:")

    st.markdown('''
    - **PBI Dashboard**: Un tablero de control creado en Python que presenta gráficos simples relacionados con el Producto Interno Bruto (PIB) a nivel mundial y regional. El objetivo principal es demostrar cómo se puede crear un tablero de control simple y efectivo utilizando Python y generando visualizaciones con Plotly. [Ver repositorio](https://github.com/fowardelcac/PBI)
    
    - **Abandono Escolar**: El proyecto se inicia con la recopilación y exploración de datos del conjunto de datos de estudiantes universitarios. Este conjunto de datos contiene información detallada sobre características de los estudiantes, como género, edad al inscribirse, becas, área de estudio y su historial académico, que incluye la cantidad de materias inscritas y aprobadas en semestres anteriores. Una vez que se han recopilado los datos y se ha realizado una limpieza y transformación adecuada, se procede a crear un modelo de Machine Learning. En este caso, se emplea el algoritmo Random Forest, un método de aprendizaje supervisado eficaz para la clasificación de estudiantes en dos categorías principales: aquellos que se graduarán con éxito y aquellos que podrían estar en riesgo de abandonar la universidad [Ver repositorio](https://github.com/fowardelcac/Abandono-Escolar/tree/main)

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
