import streamlit as st

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

from sklearn.ensemble import RandomForestClassifier

carreras = {
    "Tecnologías de Producción de Biocombustibles": 'Course_1',
    "Animación y Diseño Multimedia": 'Course_2',
    "Servicio Social (Asistencia Nocturna)": 'Course_3',
    "Agronomía": 'Course_4',
    "Diseño de Comunicación": 'Course_5',
    "Enfermería Veterinaria": 'Course_6',
    "Ingeniería Informática": 'Course_7',
    "Equinicultura": 'Course_8',
    "Gestión": 'Course_9',
    "Servicio Social": 'Course_10',
    "Turismo": 'Course_11',
    "Enfermería": 'Course_12',
    "Higiene Bucodental": 'Course_13',
    "Gestión de Publicidad y Marketing": 'Course_14',
    "Periodismo y Comunicación": 'Course_15',
    "Educación Básica": 'Course_16',
    "Gestión (Asistencia Nocturna)": 'Course_17'
}

def split(df):
  X = df.drop('Target', axis = 1)
  y = df.Target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)
  return X_train, X_test, y_train, y_test

def metricas(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  return accuracy, precision

def tasar(approved, enrolled):
    try:
        return (approved / enrolled) * 100
    except:
        return 0
@st.cache_resource
def modelo_rf(data):
    scaler = StandardScaler()
    colum_s= ['Age at enrollment', 'Tasa_1st', 'Tasa_2nd']
    data[colum_s] = scaler.fit_transform(data[colum_s])
    X_train, X_test, y_train, y_test = split(data)
    modelo = RandomForestClassifier().fit(X_train, y_train)   
    return scaler, modelo

df_course = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/Abandono-Escolar/main/Dataset/dataset.csv')
data = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/Abandono-Escolar/main/Dataset/D2/df.csv').drop('Unnamed: 0', axis = 1)

df = data.filter(['Gender','scholarship', 'Target',  'Age at enrollment', 'Tasa_1st', 'Tasa_2nd'], axis = 1)
df['Course'] = df_course.Course
df = df[df.Target != 2].copy()
df = pd.get_dummies(df, columns = ['Course'])

st.title('Abandono escolar.')

with st.expander('Modelo.'):
    scaler, modelo = modelo_rf(df)
    st.write("### Modelo Random Forest para Predicción Estudiantil")
    st.write("El modelo Random Forest es un algoritmo de aprendizaje supervisado utilizado para predecir el rendimiento académico de los estudiantes. Utiliza varios atributos como predictores para realizar estas predicciones. A continuación, se enumeran los principales atributos utilizados en el modelo:")
    st.write("1. **Sexo:** Indica el género del estudiante (Masculino o Femenino).")
    st.write("2. **Becas:** Se refiere a si el estudiante recibe becas o no.")
    st.write("3. **Edad al Momento de la Inscripción:** Representa la edad del estudiante cuando se inscribió.")
    st.write("4. **Tasa de Aprobación:** Es el cociente entre el número de materias aprobadas y el número total de materias inscritas, calculado para el primer y segundo semestre.")
    
    st.write("Este modelo utiliza estos atributos para predecir el desempeño académico de los estudiantes.")
    sexo = st.radio('Seleccionar el sexo:', ['Mujer', 'Hombre'])
    genero = 0 if sexo == 'Mujer' else 1
    
    becas = st.radio('¿Recibió alguna beca?', ['Si', 'No'])
    becario = 1 if becas == 'Si' else 0
    
    edad = st.slider('Edad de inscripción:', min_value=17, max_value=99)
    
    # TASA DE APROBACION / INSCIRPCION
    Enrolled1 = st.number_input('Inscripciones realizadas en el primer semestre:', min_value = 0)
    Approved1 = st.number_input('Materias aprobadas en el primer semestre:', min_value = 0, max_value = Enrolled1)
    Enrolled2 = st.number_input('Inscripciones realizadas en el segundo semestre:', min_value = 0)
    Approved2 = st.number_input('Materias aprobadas en el segundo semestre:', min_value = 0, max_value = Enrolled2)
    tasa1 = tasar(Approved1, Enrolled1)
    tasa2 = tasar(Approved2,Enrolled2)
    
    # SELECCIONAR MATERIA
    carrera = st.selectbox('Selecciona la carrera:', carreras.keys())
    
    predictor = pd.DataFrame(columns=['Gender', 'scholarship', 'Age at enrollment', 'Tasa_1st', 'Tasa_2nd'])
    columnas_p_numerico = ['Course_{}'.format(i) for i in range(1, 18)]
    predictor_n = pd.DataFrame(0, index=range(1), columns = columnas_p_numerico)
    
    predictor.loc[0, 'Gender'] = genero
    predictor.loc[0, 'scholarship'] = becario
    predictor.loc[0, 'Age at enrollment'] = edad
    predictor.loc[0, 'Tasa_1st'] = tasa1
    predictor.loc[0, 'Tasa_2nd'] = tasa2
    
    colum_s = ['Age at enrollment', 'Tasa_1st', 'Tasa_2nd']
    input_data = pd.concat([predictor, predictor_n], axis=1)
    input_data.loc[0, carreras[carrera]] = 1
    
    # Escala los datos de entrada del usuario
    input_data[colum_s] = scaler.transform(input_data[colum_s])
    # Realiza la predicción
    y_pred = modelo.predict(input_data)

    # Muestra el resultado de la predicción
    if y_pred[0] == 0:
        st.warning('El modelo predice que el estudiante tiene un alto riesgo de abandono.', icon="⚠️")
    else:
        st.success('El modelo predice que el estudiante tiene un bajo riesgo de abandono.', icon="✅")
