import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_df():
    df = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/credit_card_approval/main/Notebook/Dataset/synthetic_credit_card_approval.csv')
    return df

@st.cache_data
def split(df):
    X = df.drop('Target', axis=1).values.reshape((-1, 5))
    y = df.Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

@st.cache_resource
def create_model(X_train, y_train):
    return LGBMClassifier(max_depth=3, n_estimators=200).fit(X_train, y_train)
    
st.set_page_config(
    page_title = "Tarjeta",
    page_icon = "💳",
)

st.title('Aprobacion de tarjeta de credito')
df = load_df()
scaler_child = MinMaxScaler()
scaler_income = MinMaxScaler()

df.Num_Children = scaler_child.fit_transform(df['Num_Children'].values.reshape(-1 ,1))
df.Income = scaler_income.fit_transform(df['Income'].values.reshape(-1 ,1))

X_train, X_test, y_train, y_test = split(df)


lgbm = create_model(X_train, y_train)

num_child = st.slider('Ingrese la cantidad de hijos:', min_value = 0, max_value = 11)
income = st.number_input('Ingrese sus ingresos mensuales: ')

num_child = scaler_child.transform([[num_child]])
income = scaler_income.transform([[income]])

group = st.radio('Ingrese el sexo (M o F):', ('M', 'F'))
group_n = 0 if group == 'M' else 1

own_car = st.radio('¿Tiene al menos un automóvil?', ('Sí', 'No'))
own_car_n = 1 if own_car == 'Sí' else 0

own_house = st.radio('¿Posee al menos una propiedad?', ('Sí', 'No'))
own_house_n = 1 if own_house == 'Sí' else 0

try: 
    y_pred = lgbm.predict([[num_child, group_n, income, own_car_n, own_house_n]])   
    if y_pred == 0:
        st.error('Su solicitud ha sido rechazada.')       
    else:
        st.success('Felicitaciones!! Su solicitud ha sido aprobada.')
except:
    st.error('Complete los datos!')
