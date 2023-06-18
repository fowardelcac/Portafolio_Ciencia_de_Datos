import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import scipy.optimize as sci_opt

def chart(max_sh, min_vol, max_ret, df_portafolio):
    fig = px.scatter(
        df_portafolio, x='Volatibilidad', y='Retornos', color='Sharpe',
        labels={'Volatibilidad': 'Volatilidad esperada', 'Retornos': 'Retornos esperados'},
        title='Optimización de Portfolio'
    )

    fig.add_scatter(
        x=[min_vol[1]], y=[min_vol[0]], mode='markers',
        marker=dict(color='black', symbol='star', size=20), name='Máxima volatilidad esperada')
    fig.add_scatter(
        x=[max_ret[1]], y=[max_ret[0]], mode='markers',
        marker=dict(color='black', symbol='star', size=20), name='Máximos rendimientos esperados')
    fig.add_scatter(
        x=[max_sh[1]], y=[max_sh[0]], mode='markers',
        marker=dict(color='black', symbol='star', size=20), name='Máximo Sharpe Ratio')

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

def benchmark(df, sp, pesos:list):
    cap = 100000
    data = df / df.iloc[0]
    dff = pd.DataFrame()
    indice = 0
    for i in data:
        dff[i] = (data[i] * pesos[indice]) * cap
        indice += 1
    
    
    dff['Value'] = dff.sum(axis=1)
    dff['SP500'] = (sp / sp.iloc[0]) * cap
    dff.dropna(inplace=True)
    return dff

def calculos_(weights: list):
    ret = np.sum(df_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(df_ret.cov() * 252, weights)))
    sr = (ret - 0.01) / vol
    return np.array([ret, vol, sr])

def neg_s(weights: list) -> np.array:
    return calculos_(weights)[2] * (-1)


st.set_page_config(
    page_title="Portfolio",
    page_icon=":open_file_folder:",
)

assets = ['AAPL', 'MSFT', 'TSLA']
n_stocks = len(assets)
assets_df = []
for i in assets:
    assets_df.append(yf.download(i, '2015-01-01')['Adj Close']) 
df = pd.concat(assets_df, axis=1)
df.columns = assets
df.dropna(inplace=True)
sp = yf.download('^GSPC', '2015-01-01')['Adj Close']
df_ret = np.log(df / df.shift(1)).dropna()
option = st.selectbox('¿Cómo le gustaría obtener su cartera?', ('Simulación de Monte Carlo', 'Optimización por Sharpe ratio'))

n_iter = 5000
portfolio_returns, portfolio_volatilities, portfolio_sharpe = [], [], []
all_weights = np.zeros((n_iter, n_stocks))

for i in range(n_iter):
    weights = np.random.random(n_stocks)
    weights = weights / np.sum(weights)

    all_weights[i, :] = weights
    ret = np.sum(df_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(df_ret.cov() * 252, weights)))
    sr = (ret-0.01) / vol

    portfolio_returns.append(ret)
    portfolio_volatilities.append(vol)
    portfolio_sharpe.append(sr)

df_portafolio = pd.DataFrame({
    'Retornos': portfolio_returns,
    'Volatibilidad': portfolio_volatilities,
    'Sharpe': portfolio_sharpe,
    'Pesos': np.round(all_weights, 3).tolist()
})

st.write(df_portafolio)
max_sh = df_portafolio.iloc[df_portafolio.Sharpe.idxmax()]
min_vol = df_portafolio.iloc[df_portafolio.Volatibilidad.idxmin()]
max_ret = df_portafolio.iloc[df_portafolio.Retornos.idxmax()]
chart(max_sh, min_vol, max_ret, df_portafolio)

opt_pesos = st.selectbox('Elegir pesos:', df_portafolio.Pesos)
st.subheader('Benchmark:')
bench = benchmark(df, sp, opt_pesos)
st.write(bench)
rdo_df = bench.filter(['Value', 'SP500'], axis=1)
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
    
st.subheader('Optimización del portafolio sobre el Sharpe ratio')
bounds = tuple((0, 1) for symbol in range(n_stocks))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
init_guess = n_stocks * [1 / n_stocks]

optimized_sharpe = sci_opt.minimize(
    neg_s,
    init_guess,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimized_metrics = calculos_(weights=optimized_sharpe.x)
st.subheader('Resultados:')

result = pd.DataFrame({
    'Retornos': optimized_metrics[0],
    'Volatilidad': optimized_metrics[1],
    'Sharpe': optimized_metrics[2],
    'Pesos': [np.round(optimized_sharpe.x, 3)]
})
st.write(result)

st.subheader("Benchmark")
bench = benchmark(df, sp, result.Pesos)
st.write(bench)
rdo_df = bench.filter(['Value', 'SP500'], axis=1)
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
           
