import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title = "PBI Dashboard", page_icon = ":bar_chart:", layout = "wide")

g20 = [
    'Argentina',
    'Australia',
    'Brazil',
    'Canada',
    'China',
    'France',
    'Germany',
    'India',
    'Indonesia',
    'Italy',
    'Japan',
    'Mexico',
    'Russia',
    'Saudi Arabia',
    'South Africa',
    'South Korea',
    'Turkey',
    'United Kingdom',
    'United States',
]

# Países miembros del G7 en inglés
g7 = [
    'Canada',
    'France',
    'Germany',
    'Italy',
    'Japan',
    'United Kingdom',
    'United States',
]

# Países miembros del BRICS en inglés
brics = [
    'Brazil',
    'Russia',
    'India',
    'China',
    'South Africa',
]

@st.cache_data
def cargar_datos():        
    gdp1 = pd.read_csv('https://github.com/fowardelcac/PBI/raw/main/PBI_dataset/Originales/gdp_data.csv', delimiter=',')
    gdp2 = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/PBI/main/PBI_dataset/Originales/country_codes.csv', delimiter=',' )
    return gdp1, gdp2

@st.cache_data
def merge_gdps(df1, df2):
    return pd.merge(df1, df2, on = 'country_code', how = 'inner')

@st.cache_data
def separar_grupos(df):
    brics_df = df[df['country_name'].isin(brics)].set_index('year')
    g20_df = df[df['country_name'].isin(g20)].set_index('year')
    g7_df = df[df['country_name'].isin(g7)].set_index('year')
    return brics_df, g20_df, g7_df

@st.cache_data
def agrupar_region_22(df):
  data_agrup_cont = df.groupby(['region','year'])['value'].sum().to_frame()
  dff = data_agrup_cont.reset_index(['region', 'year'])
  dff1 = dff[dff.year==2022]
  return dff1

def graf_pbi_mundial(df):
  mundial = df.groupby('year').sum(numeric_only='True')
  fig = px.line(mundial, mundial.index, 'value')
  fig.update_layout(
      xaxis_title='Año',
      yaxis_title='Valor del PBI'
  )
  return fig

def crecimiento_por_pais(country, df):
  df = df[df['country_name'] == country].set_index('year')

  fig = px.line(df, df.index, 'value')
  fig.update_layout(
      title = f'Crecimiento en {country} a lo largo de los años',
      xaxis_title = 'Año',
      yaxis_title = 'Valor del PBI'
  )
  return fig

def pie_por_grupo(grupo, texto):
  '''
  grupo: dataframe del grupo
  texto: nombre del grupo, g20...
  '''
  ultimo_a = grupo[grupo.index == 2022]
  fig = px.pie(ultimo_a, values='value', names='country_code')
  fig.update_layout(
      title = f'Integrantes del {texto} y su PBI',
      xaxis_title = 'Año',
      yaxis_title = 'Valor del PBI'
  )
  return fig

def pbi_por_continente_barra(df):
  dff1 = agrupar_region_22(df)

  fig = px.bar(dff1, x='region', y='value')
  fig.update_layout(
      title = 'PBI por region.'
  )
  return fig

def pbi_porc_pie(df):
  df = agrupar_region_22(df)
  fig = px.pie(df, values='value', names='region')
  fig.update_layout(
      xaxis_title = 'Año',
      yaxis_title = 'Valor del PBI'
  )
  fig.update_traces(texttemplate='', textposition='outside')
  return fig

def pbi_hist_regi(df):
  data_agrup_cont = df.groupby(['region','year'])['value'].sum().to_frame()
  fig = px.line(data_agrup_cont.reset_index(), x='year', y='value', color='region',
                labels={'year': 'Año', 'value': 'Valor', 'region': 'Región'},
                title='Evolución del PBI por Región')

  fig.update_layout(legend=dict(x=1, y=1.2))  # Coloca la leyenda fuera del gráfico
  return fig

@st.cache_data
def mapa(df):
  pais_index = df.loc[df.groupby('country_name')['year'].idxmax()].filter(['country_name', 'value', 'year', 'country_code'], axis = 1)

  fig = px.choropleth(
      pais_index,
      locations = 'country_code',  # Columna con códigos ISO de país
      color= 'value',          # Columna con los valores para la coloración
      hover_name = 'country_name',   # Columna con nombres de país para información al pasar el mouse
      color_continuous_scale = 'plasma'  # Escala de colores
    )

  fig.update_geos(
        resolution=50,
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="lightblue",
      )
  return fig

df1, df2 = cargar_datos()
df = merge_gdps(df1, df2)
BRICS, G20, G7 = separar_grupos(df)
dic_var = {
    'G20': G20,
    'G7': G7,
    'BRICS': BRICS
    }

col1, col2 = st.columns([2, 1], gap = 'small')
col11, col22 = st.columns([2, 1], gap = 'small')
col13, col23 = st.columns(2, gap = 'small')


with col1:
    st.title('Mapa del PBI')
    st.plotly_chart(mapa(df))
   
    
with col2:
    st.title('PBI por region')
    st.plotly_chart(pbi_porc_pie(df))
    #

with col11:
    op = st.selectbox('Seleccione un pais:', df.country_name.unique())
    st.plotly_chart(crecimiento_por_pais(op, df))

with col22:
    grupo = st.selectbox('Seleccione un grupo:', ('G20', 'G7', 'BRICS'))
    st.plotly_chart(pie_por_grupo(dic_var[grupo], grupo))

with col13:
    st.title('PBI mundial desde 1960')
    st.plotly_chart(graf_pbi_mundial(df))
with col23:
    st.plotly_chart(pbi_hist_regi(df))

