import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.optimize as sci_opt

st.set_page_config(
    page_title="Portfolio",
    page_icon=":open_file_folder:",
)

def chart_frontera(df_frontera, max_sh, min_vol, max_ret):
  fig = px.scatter(df_frontera, x='Volatibilidad', y='Retornos', color='Sharpe',
                 labels={'Volatibilidad': 'Volatibilidad esperada', 'Retornos': 'Retornos esperados'},
                 title='Portfolio Optimization',
                 hover_data={'Volatibilidad': True, 'Retornos': True, 'Sharpe': True})

  fig.add_scatter(x=[min_vol[1]], y=[min_vol[0]], mode='markers', marker=dict(color='red', symbol='star', size=20), name='Volatibilidad minima esperada')
  fig.add_scatter(x=[max_ret[1]], y=[max_ret[0]], mode='markers', marker=dict(color='red', symbol='star', size=20), name='Retornos maximos esperados')
  fig.add_scatter(x=[max_sh[1]], y=[max_sh[0]], mode='markers', marker=dict(color='red', symbol='square', size=20), name='Maximo Sharpe Ratio')

  fig.update_layout(
      coloraxis=dict(colorscale='plasma', colorbar=dict(title='Sharpe Ratio')),
      legend=dict(
          title=None,
          orientation='h',
          yanchor='bottom',
          y=-0.2,
          xanchor='right',
          x=1
      ),
      margin=dict(l=0, r=0, t=30, b=0),
      showlegend=True,
      hovermode='closest',
      plot_bgcolor='white',
      paper_bgcolor='white',
      xaxis=dict(showgrid=True, title_font=dict(size=12)),
      yaxis=dict(showgrid=True, title_font=dict(size=12)),
  )

  st.subheader("Frontera eficiente de Markowitz")
  st.plotly_chart(fig, use_container_width=True)

def benchmark(cap, pesos, df):
  data1 = df.drop(['Date', '^GSPC'], axis=1)
  sp = df['^GSPC']
  data1 = data1 / data1.iloc[0]
  
  dff = pd.DataFrame()
  indice = 0
  for i in data1:
      dff[i] = (data1[i] * pesos[indice]) * cap
      indice += 1

  dff['Date'] = df.Date
  dff['Value'] = dff.sum(axis=1, numeric_only=True)
  dff['SP500'] = (sp / sp.iloc[0]) * cap
  return dff.dropna()

st.title('Optimizacion de cartera.')

assets_df = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/Portafolio_Ciencia_de_Datos/main/assets_df.csv')
df_ret = pd.read_csv('https://raw.githubusercontent.com/fowardelcac/Portafolio_Ciencia_de_Datos/main/df_ret.csv')

n_iter, n_assets = 5000, 3
pf_returns, pf_vol, pf_sharpe = [list() for _ in range(3)]
all_weights = np.zeros((n_iter, n_assets))


for i in range(n_iter):
  weights = np.random.random(n_assets)

  weights = weights / np.sum(weights)
  all_weights[i, :] = weights

  ret_esp = np.sum(df_ret.mean(numeric_only=True) * weights) * 252
  vol_esp = np.sqrt(np.dot(weights.T, np.dot(df_ret.cov(numeric_only=True) * 252, weights)))
  sharpe = (ret_esp - 0.01) / vol_esp
  pf_returns.append(ret_esp)
  pf_vol.append(vol_esp)
  pf_sharpe.append(sharpe)

df_frontera = pd.DataFrame({
    'Retornos': pf_returns,
    'Volatibilidad': pf_vol,
    'Sharpe': pf_sharpe,
    'Pesos': np.round(all_weights, 3).tolist()
})

st.title('Simulaciones de Monte Carlo')
st.write(df_frontera)

max_sh = df_frontera.iloc[df_frontera.Sharpe.idxmax()]
min_vol = df_frontera.iloc[df_frontera.Volatibilidad.idxmin()]
max_ret = df_frontera.iloc[df_frontera.Retornos.idxmax()]
chart_frontera(df_frontera, max_sh, min_vol, max_ret)
     
st.subheader('Benchmark: Portafolio vs SP500')
st.text('Se seleccionaron los pesos de acuerdo al mejor retorno.')
bench_sharpe = benchmark(100000, max_ret.Pesos, assets_df)
rdo_df = bench_sharpe.filter(['Value', 'SP500'], axis=1)
st.write(rdo_df)

rdo_last = round(rdo_df.iloc[-1], 3)

rdo_ret = (np.log(rdo_df / rdo_df.shift(1))).dropna()
rdo_pd = pd.DataFrame({
    'Retorno medio anual': round((rdo_ret[['Value', 'SP500']].mean() * 250) * 100, 2),
    'Vol. media anual': round((rdo_ret[['Value', 'SP500']].std() * 250 ** 0.5) * 100, 2)
})

st.write('-' * 100)
st.text('Si hubieras invertido $100,000: ')
st.text(f"Portafolio: ${rdo_last.Value}")
st.text(f"SP500: ${rdo_last.SP500}")
st.write('-' * 100)

st.text('Rendimiento vs volatibilidad anualizada:')
st.write(rdo_pd)
st.write('-' * 100)
st.line_chart(rdo_df)
     
def calculos_(weights: list):
    ret = np.sum(df_ret.mean(numeric_only=True) * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(df_ret.cov(numeric_only=True) * 252, weights)))
    sr = (ret - 0.01) / vol
    return np.array([ret, vol, sr])

def neg_s(weights: list) -> np.array:
    return calculos_(weights)[2] * (-1)

st.title('Optimización del Sharpe ratio')
bounds = tuple((0, 1) for symbol in range(n_assets))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
init_guess = n_assets * [1 / n_assets]

optimized_sharpe = sci_opt.minimize(
    neg_s,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimized_metrics = calculos_(weights=optimized_sharpe.x)

result = pd.DataFrame({
    'Retornos': optimized_metrics[0],
    'Volatilidad': optimized_metrics[1],
    'Sharpe': optimized_metrics[2],
    'Pesos': [np.round(optimized_sharpe.x, 3).tolist()]
})
st.subheader('Resultados:')

st.write(result)

st.subheader('Benchmark: Portafolio vs SP500')
bench = benchmark(100000, result.Pesos[0], assets_df)

rdo_df = bench.filter(['Value', 'SP500'], axis=1)
rdo_last = round(rdo_df.iloc[-1], 3)

rdo_ret = (np.log(rdo_df / rdo_df.shift(1))).dropna()
rdo_pd = pd.DataFrame({
    'Retorno medio anual': round((rdo_ret[['Value', 'SP500']].mean(numeric_only=True) * 250) * 100, 2),
    'Vol. media anual': round((rdo_ret[['Value', 'SP500']].std(numeric_only=True) * 250 ** 0.5) * 100, 2)
})

st.write(rdo_pd)

st.write('-' * 100)
st.text('Si hubieras invertido $100,000: ')
st.text(f"Portafolio: ${rdo_last.Value}")
st.text(f"SP500: ${rdo_last.SP500}")
st.write('-' * 100)

st.text('Rendimiento vs volatibilidad anualizada:')
st.write(rdo_pd)
st.write('-' * 100)
st.line_chart(rdo_df)
