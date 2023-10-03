import pandas as pd
import plotly.express as px
import streamlit as st

import listas as l

st.set_page_config(page_title = "PBI Dashboard", page_icon = ":bar_chart:", layout = "wide")


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
    brics_df = df[df['country_name'].isin(l.brics)].set_index('year')
    g20_df = df[df['country_name'].isin(l.g20)].set_index('year')
    g7_df = df[df['country_name'].isin(l.g7)].set_index('year')
    return brics_df, g20_df, g7_df

@st.cache_data
def agrupar_region_22(df):
  data_agrup_cont = df.groupby(['region','year'])['value'].sum().to_frame()
  dff = data_agrup_cont.reset_index(['region', 'year'])
  dff1 = dff[dff.year==2022]
  return dff1

@st.cache_data
def graf_pbi_mundial(df):
  fig = px.line(df, df.index, 'value')
  fig.update_layout(
      xaxis_title='Año',
      yaxis_title='Valor del PBI',
      width = 600,   # Ancho personalizado
      height= 400
  )
  return fig

@st.cache_data
def crecimiento_por_pais(country, df):
  df = df[df['country_name'] == country].set_index('year')

  fig = px.line(df, df.index, 'value')
  fig.update_layout(
      title = f'Evolucion en {country} a lo largo de los años',
      xaxis_title = 'Año',
      yaxis_title = 'Valor del PBI'
  )
  return fig

@st.cache_data
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
      yaxis_title = 'Valor del PBI',
      width = 500,   # Ancho personalizado
      height= 500
  )
  return fig

@st.cache_data
def pbi_por_continente_barra(df):
  dff1 = agrupar_region_22(df)
  fig = px.bar(dff1, x='region', y='value')
  fig.update_layout(
      xaxis_title = 'Region',
      yaxis_title = 'Valor del PBI',
      width = 600,   # Ancho personalizado
      height= 500
  )
  return fig

@st.cache_data
def pbi_porc_pie(df):
  df = agrupar_region_22(df)
  fig = px.pie(df, values='value', names='region')
  fig.update_layout(
      xaxis_title = 'Año',
      yaxis_title = 'Valor del PBI'
  )
  fig.update_traces(texttemplate='', textposition='outside')
  return fig

@st.cache_data
def pbi_hist_regi(df):
  data_agrup_cont = df.groupby(['region','year'])['value'].sum().to_frame()
  fig = px.line(data_agrup_cont.reset_index(), x='year', y='value', color='region',
                labels={'year': 'Año', 'value': 'Valor', 'region': 'Región'})
  fig.update_layout(
      legend=dict(x=1, y=1.2),
      width = 900,   # Ancho personalizado
      height= 500
      )  # Coloca la leyenda fuera del gráfico
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
  fig.update_layout(
      width = 600,   # Ancho personalizado
      height= 400
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

st.title("PBI Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.title('Mapa del PBI')
    st.plotly_chart(mapa(df))
    st.header('PBI por region(2022)')
    st.plotly_chart(pbi_por_continente_barra(df))

   
with col2:
    st.header('Evolucion del PBI')
    mundial = df.groupby('year').sum(numeric_only='True')
    fecha = st.slider('Rango de años.', min_value = mundial.index.min(), max_value = mundial.index.max(), value = (mundial.index.min(), mundial.index.max()))
    mundial_filtrado = mundial.loc[fecha[0]:fecha[1]]
    st.plotly_chart(graf_pbi_mundial(mundial_filtrado))
    
    st.header('Foros internacionales.')
    grupo = st.selectbox('Seleccione un grupo:', ('G20', 'G7', 'BRICS'))
    st.plotly_chart(pie_por_grupo(dic_var[grupo], grupo))

with st.expander('Variacion por pais.'):
    st.write('Permite seleccionar un pais y como evoluciono su PBI.')
    op = st.selectbox('Seleccione un pais:', df.country_name.unique())
    st.plotly_chart(crecimiento_por_pais(op, df))

with st.expander('Evolucion por region.'):
    st.plotly_chart(pbi_hist_regi(df))

with st.expander('Dataset'):
    st.table(df)