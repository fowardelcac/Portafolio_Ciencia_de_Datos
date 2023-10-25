import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Análisis de Deserción y Éxito Académico', layout='wide')
st.title('Análisis de Deserción y Éxito Académico de Estudiantes Universitarios')
st.set_option('deprecation.showPyplotGlobalUse', False)

carreras = {
    "Tecnologías de Producción de Biocombustibles": 'Carrera_1',
    "Animación y Diseño Multimedia": 'Carrera_2',
    "Servicio Social (Asistencia Nocturna)": 'Carrera_3',
    "Agronomía": 'Carrera_4',
    "Diseño de Comunicación": 'Carrera_5',
    "Enfermería Veterinaria": 'Carrera_6',
    "Ingeniería Informática": 'Carrera_7',
    "Equinicultura": 'Carrera_8',
    "Gestión": 'Carrera_9',
    "Servicio Social": 'Carrera_10',
    "Turismo": 'Carrera_11',
    "Enfermería": 'Carrera_12',
    "Higiene Bucodental": 'Carrera_13',
    "Gestión de Publicidad y Marketing": 'Carrera_14',
    "Periodismo y Comunicación": 'Carrera_15',
    "Educación Básica": 'Carrera_16',
    "Gestión (Asistencia Nocturna)": 'Carrera_17'
}

def cargar_datos():
  data = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/Abandono-Escolar/main/Dataset/dataset.csv')
  return data.filter(['Course', 'International',
       'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (approved)', 'Target'], axis = 1)

def procesar_datos(data):
    df = pd.DataFrame({
        'Becarios': data['Scholarship holder'],
        'Inscritos1': data['Curricular units 1st sem (enrolled)'],
        'Aprobados1': data['Curricular units 1st sem (approved)'],
        'Inscritos2': data['Curricular units 2nd sem (enrolled)'],
        'Aprobados2': data['Curricular units 2nd sem (approved)'],
        'Género': data['Gender'],
        'Edad al Inscribirse': data['Age at enrollment'],
        'Internacional': data['International'],
        'Carrera': data['Course']
    })
    df['Estado'] = data['Target'].map({'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}).copy()
    categorias = {
      'Tecnología y Diseño': [2, 5, 7],
      'Agronomía y Ciencias Veterinarias': [4, 6, 8],
      'Servicios Sociales y Salud': [3, 10, 12, 13],
      'Negocios y Comunicación': [5, 9, 17, 14, 15],
      'Turismo': [11],
      'Educación': [16],
      'Energías Renovables': [1]
    }

    df['Área de Estudio'] = df['Carrera'].apply(lambda x: next((area for area, carreras in categorias.items() if x in carreras), None))
    df = df[df['Estado'] != 2]
    df['Tasa_1st'] = tasar(df['Inscritos1'], df['Aprobados1'])
    df['Tasa_2nd'] = tasar(df['Inscritos2'], df['Aprobados2'])
    return df

def frecuencia_relativa(muestra, subconjunto):
    N = len(muestra)
    return ((len(subconjunto) / N) * 100)

def split(df):
  X = df.drop('Estado', axis = 1)
  y = df.Estado
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)
  return X_train, X_test, y_train, y_test

def tasar(Inscritos, Aprobados):
    try:
        return (Aprobados / Inscritos) * 100
    except:
        return 0
@st.cache_resource
def modelo_rf(data):
    scaler = StandardScaler()
    colum_s= ['Edad al Inscribirse', 'Tasa_1st', 'Tasa_2nd']
    data[colum_s] = scaler.fit_transform(data[colum_s])
    X_train, X_test, y_train, y_test = split(data)
    modelo = RandomForestClassifier().fit(X_train, y_train)   
    return scaler, modelo

@st.cache_data
def analisis(df):
    data_becarios = df[df['Becarios'] == 1]
    dic = (df['Área de Estudio'].value_counts()).to_dict()
    st.write('En este proyecto, exploramos un conjunto de datos centrado en el rendimiento académico de estudiantes universitarios. Nuestro objetivo es comprender las razones detrás de la deserción y la graduación. El conjunto de datos revela tres estados principales: abandono (32.12%), estudiantes cursando (17.94%) y graduados (49.9%). Nos enfocaremos en las variables relacionadas con abandono y graduación.')    
    with st.expander('Distribución de la Variable Objetivo'):
        plt.figure(figsize=(8, 4))
        sns.countplot(data = df, x = 'Estado', palette = {0: "red", 1: "green"})
        plt.title("Grafico de barras sobre la variable objetivo.")
        plt.xlabel("Estado")
        plt.xticks(ticks=[0,1], labels=["Abandondo", "Graduado"])
        st.pyplot()
    st.subheader('Perfil de los Estudiantes')
    st.write('Basándonos en nuestro análisis, observamos que el 75% de los estudiantes se inscriben entre los 17 y 25 años. La proporción de hombres en la muestra es del 34.41%, mientras que la de mujeres es del 65.59%. En cada grupo de 100 estudiantes graduados, encontramos 24.81 hombres y 75.19 mujeres, lo que destaca una significativa brecha de género.')

    with st.expander('Distribución de Edades'):
        plt.figure(figsize=(12, 4))
        sns.histplot(df['Edad al Inscribirse'], kde=True, color='blue')
        plt.xlabel('Edad al momento de la inscripción')
        plt.ylabel('Frecuencias')
        plt.title('Edad al momento de la inscripción')
        st.pyplot()
        
    with st.expander('Distribución de Género'):
        plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(121)  # 1 fila, 2 columnas, primer subplot
        sns.countplot(x='Género', data=df, ax=ax1)
        ax1.set_title("Sexo masculino vs femenino")
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Femenino', 'Masculino'])

        ax2 = plt.subplot(122)  # 1 fila, 2 columnas, segundo subplot
        sns.countplot(data=df, x='Género', hue='Estado', palette={0: "red", 1: "green"}, ax=ax2)
        ax2.set_title("Estado de cursado según el sexo")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Femenino', 'Masculino'])
        st.pyplot()

        h = frecuencia_relativa(df, df[df['Género'] == 1])
        m = frecuencia_relativa(df, df[df['Género'] == 0])
        st.write(f'La proporción de hombres en la muestra es del {h:.2f}%, mientras que las mujeres representan el {m:.2f}%. Cada 100 estudiantes graduados incluyen 24.81 hombres y 75.19 mujeres.')
    st.subheader('Análisis por Áreas de Estudio')
    st.write('Para facilitar el análisis, he agrupado las carreras en distintas áreas:')

    with st.expander('Distribución de sexos por Área'):
        plt.figure(figsize=(22, 6))
        sns.countplot(data=df, x='Área de Estudio', hue='Género', palette={0: 'pink', 1: 'blue'})
        plt.title('Tipo de carrera elegida según el sexo')
        plt.xlabel('Tipo de carrera')
        plt.ylabel('Cantidad')
        plt.legend(title='Sexo', labels=['Femenino', 'Masculino'])
        plt.xticks(ticks=range(len(dic)), labels=dic.keys())
        st.pyplot()
        
        st.write('Las mujeres tienden a centrarse en dos areas: "Tecnologia y diseño" y "Agronomia y ciencias veterinarias"')
        
        plt.figure(figsize=(22, 6))
        sns.countplot(data = df, x ='Área de Estudio', hue = 'Estado',  palette = {0: "red", 1: "green"})
        plt.title("Estado de cursado segun el sexo.")
        plt.xlabel("Tipo de carrera")
        plt.ylabel("Estado")
        plt.legend(title = "Estado", labels=["Abandono", "Graduado"])
        plt.xticks(ticks=range(len(dic)), labels=dic.keys())
        st.pyplot()
        st.write("Se puede apreciar que los dos sectores 'Tecnologia y diseño' y 'Agronomia y ciencias veterinarias' son las carreras con mas graduados, y si observamos en el primer grafico vemos que ambas carreras cuentran con un mayor cursado de mujeres, podriamos suponer que esto podria generar una influencia en el porque las mujeres se graduan tres veces mas que los hombres")
    
    st.subheader('Becas')
    st.write('En nuestra universidad, 969 estudiantes están cursando gracias a becas universitarias, y la mayoría son mujeres. Notablemente, el país de origen no parece influir en la distribución de becas. La universidad financia principalmente dos áreas: Turismo y Servicios Sociales y Salud, la cual incluye: Servicio Social, Enfermería y Higiene Oral. Además, es evidente que los estudiantes que han recibido becas tienen una tasa de éxito significativamente mayor.')

    with st.expander('Estudiantes que reciben becas comparados por sexo y nacionalidad'):
        plt.subplot(1, 2, 1)  # Primer subplot
        plt.title('Estudiantes que reciben becas comparados por sexo')
        sns.countplot(data=data_becarios, x='Becarios', hue='Género', palette={0: 'pink', 1: 'blue'})
        plt.legend(title='Sexo', labels=['Femenino', 'Masculino'])
        plt.xlabel('Becarios')
        plt.xticks(ticks=[0], labels=[])
        
        plt.subplot(1, 2, 2)  # Segundo subplot
        plt.title('Estudiantes que reciben becas comparados por nacionalidad')
        sns.countplot(data=data_becarios, x='Becarios', hue='Internacional', palette={0: 'blue', 1: 'yellow'})
        plt.legend(title='Nacionalidad', labels=['Nativo', 'Extranjero'])
        plt.xlabel('Becarios')
        plt.xticks(ticks=[0], labels=[])
        st.pyplot()
    
    with st.expander('Tipo de carrera más subsidiada'):
        plt.figure(figsize=(22, 6))
        plt.title('Tipo de carrera más subsidiada')
        sns.countplot(data=data_becarios, x='Área de Estudio', hue='Becarios', palette={1: 'green'})
        plt.xlabel('Tipo de carrera')
        plt.ylabel('Becas')
        plt.legend(title='Becas', labels=['No', 'Sí'])
        plt.xticks(ticks=range(len(dic)), labels=dic.keys())
        st.pyplot()
    
    with st.expander('Estudiantes que reciben becas vs Target'):
        data_becarios = df[df['Becarios'] == 1]
        sns.countplot(data=data_becarios, x='Becarios', hue='Estado', palette={0: 'red', 1: 'green'})
        plt.title('Estudiantes que reciben becas vs Target')
        plt.legend(title='Estado', labels=['Abandono', 'Graduado'])
        plt.xlabel('Becarios')
        plt.xticks(ticks=[0], labels=[])
        st.pyplot()
            
    st.subheader('Modelo de Aprendizaje Automático')
    st.write(
        'Como parte de final del análisis, cree un modelo de aprendizaje automático para predecir si un estudiante abandonará sus estudios o los completará con éxito. Utilice datos detallados sobre los estudiantes, incluyendo su edad, género, becas, área de estudio y más, para desarrollar este modelo. Este análisis proporciona valiosas ideas sobre los factores que influyen en el abandono y el éxito académico de los estudiantes universitarios, lo que puede ayudar a las instituciones educativas a tomar decisiones más informadas y a brindar un mejor apoyo a sus estudiantes.')

if __name__ == '__main__':
    df = procesar_datos(cargar_datos())
    analisis(df)

    data = df.filter(['Género', 'Becarios', 'Edad al Inscribirse', 'Tasa_1st', 'Tasa_2nd', 'Carrera', 'Estado'], axis = 1).fillna(0)
    df = pd.get_dummies(data, columns = ['Carrera'])  
    with st.expander('Modelo.'):
        scaler, modelo = modelo_rf(df)
        st.write("### Modelo Random Forest para Predicción Estudiantil")
        st.write("El modelo Random Forest es un algoritmo de aprendizaje supervisado utilizado para predecir el rendimiento académico de los estudiantes. Utiliza varios atributos como predictores para realizar estas predicciones. A continuación, se enumeran los principales atributos utilizados en el modelo:")
        st.write("1. **Sexo:** Indica el género del estudiante (Masculino o Femenino).")
        st.write("2. **Becas:** Se refiere a si el estudiante recibe becas o no.")
        st.write("3. **Edad al Momento de la Inscripción:** Representa la edad del estudiante cuando se inscribió.")
        st.write("4. **Tasa de Aprobación:** Es el cociente entre el número de materias aprobadas y el número total de materias inscritas, calculado para el primer y segundo semestre.")
    
        st.write("Este modelo utiliza estos atributos para predecir el desempeño académico de los estudiantes. Y cuenta con un accuracy del: 0.89 y una precision: 0.91.")
        sexo = st.radio('Seleccionar el sexo:', ['Mujer', 'Hombre'])
        genero = 0 if sexo == 'Mujer' else 1
        
        becas = st.radio('¿Recibió alguna beca?', ['Si', 'No'])
        becario = 1 if becas == 'Si' else 0
        
        edad = st.slider('Edad de inscripción:', min_value=17, max_value=99)
        
        Enrolled1 = st.number_input('Inscripciones realizadas en el primer semestre:', min_value = 0)
        Approved1 = st.number_input('Materias aprobadas en el primer semestre:', min_value = 0, max_value = Enrolled1)
        Enrolled2 = st.number_input('Inscripciones realizadas en el segundo semestre:', min_value = 0)
        Approved2 = st.number_input('Materias aprobadas en el segundo semestre:', min_value = 0, max_value = Enrolled2)
        tasa1 = tasar(Enrolled1, Approved1)
        tasa2 = tasar(Enrolled2, Approved2)
        
        carrera_ = st.selectbox('Selecciona la carrera:', carreras.keys())
        predictor = pd.DataFrame(columns=['Género', 'Becarios', 'Edad al Inscribirse', 'Tasa_1st', 'Tasa_2nd'])
        columnas_p_numerico = ['Carrera_{}'.format(i) for i in range(1, 18)]
        predictor_n = pd.DataFrame(0, index=range(1), columns = columnas_p_numerico)
        predictor.loc[0, 'Género'] = genero
        predictor.loc[0, 'Becarios'] = becario
        predictor.loc[0, 'Edad al Inscribirse'] = edad
        predictor.loc[0, 'Tasa_1st'] = tasa1
        predictor.loc[0, 'Tasa_2nd'] = tasa2
        
        colum_s = ['Edad al Inscribirse', 'Tasa_1st', 'Tasa_2nd']
        input_data = pd.concat([predictor, predictor_n], axis=1)
        input_data.loc[0, carreras[carrera_]] = 1        
        input_data[colum_s] = scaler.transform(input_data[colum_s])
        y_pred = modelo.predict(input_data)
        
        if y_pred[0] == 0:
            st.warning('El modelo predice que el estudiante tiene un alto riesgo de abandono.', icon="⚠️")
        else:
            st.success('El modelo predice que el estudiante tiene un bajo riesgo de abandono.', icon="✅")

