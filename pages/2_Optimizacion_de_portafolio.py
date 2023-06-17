import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import scipy.optimize as sci_opt

@st.cache
def descarga(lista):
    assets_df = []
    for i in lista:
        try:
            assets_df.append(yf.download(i, '2015-01-01')['Adj Close'])
        except:
            st.write("Compruebe los tickers!")
    df = pd.concat(assets_df, axis=1)
    df.columns = lista
    df.dropna(inplace=True)
    return df

@st.cache_data
def montecarlo(n_iter, n_stocks, df_retornos):
    portfolio_returns, portfolio_volatilities, portfolio_sharpe = [], [], []
    all_weights = np.zeros((n_iter, n_stocks))

    for i in range(n_iter):
        weights = np.random.random(n_stocks)
        weights = weights / np.sum(weights)

        all_weights[i, :] = weights
        ret = np.sum(df_retornos.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(df_retornos.cov() * 252, weights)))
        sr = ret / vol

        portfolio_returns.append(ret)
        portfolio_volatilities.append(vol)
        portfolio_sharpe.append(sr)

    df_portafolio = pd.DataFrame({
        'Retornos': portfolio_returns,
        'Volatibilidad': portfolio_volatilities,
        'Sharpe': portfolio_sharpe,
        'Pesos': np.round(all_weights, 3).tolist()
    })

    return df_portafolio


def benchmark(df, sp, pesos=list):
    data = df / df.iloc[0]
    dff = pd.DataFrame()
    indice = -1
    for i in data:
        indice += 1
        dff[i] = (data[i] * pesos[indice]) * 100000

    dff['Value'] = dff.sum(axis=1)
    dff['SP500'] = (sp / sp.iloc[0]) * 100000
    dff.dropna(inplace=True)

    rdo_df = dff.filter(['Value', 'SP500'], axis=1)
    rdo_last = round(rdo_df.iloc[-1], 3)

    df_ret = (np.log(rdo_df / rdo_df.shift(1))).dropna()
    rdo_pd = pd.DataFrame({
        'Retorno medio anual': round((df_ret[['Value', 'SP500']].mean() * 250) * 100, 2),
        'Vol. media anual': round((df_ret[['Value', 'SP500']].std() * 250 ** 0.5) * 100, 2)
    })

    st.write('-' * 100)
    st.text('Si hubieras invertido $100,000: ')
    st.text(f"Portafolio: ${rdo_last.Value}")
    st.text(f"SP500: ${rdo_last.SP500}")

    st.text('Rendimiento vs volatibilidad anualizada:')
    st.write(rdo_pd)
    st.write('-' * 100)
    return dff

@st.cache
def calculos_(weights: list):
    ret = np.sum(df_retornos.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(df_retornos.cov() * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])

def neg_s(weights: list) -> np.array:
    return calculos_(weights)[2] * (-1)

def get_vol(weights: list) -> np.array:
    return calculos_(weights)[1]

st.set_page_config(
    page_title="Portfolio",
    page_icon=":open_file_folder:",
)

st.number_input('Ingrese el número de activos:', min_value=0, max_value=5, key='N')
if 'N' in st.session_state and st.session_state.N > 1:
    st.text('Formato de ticker: MSFT, TSLA, AAPL, etc.')
    tickers = st.text_input('Ingrese los tickers:', key='ticker').upper()
    tickers_list = [ticker.strip() for ticker in tickers.split(',')]

    if st.session_state.N == len(tickers_list):
        if 'df' not in st.session_state:
            df = descarga(tickers_list)
            sp = descarga(['^GSPC'])
            ret_log = (np.log(df / df.shift(1))).dropna()

            st.session_state['df'] = df
            st.session_state['ret_log'] = ret_log
            st.session_state['Sp'] = sp
        st.write('--' * 100)
        option = st.selectbox(
            '¿Cómo le gustaría obtener su cartera?',
            ('Simulación de Monte Carlo', 'Optimización por Sharpe ratio', 'Optimización por volatilidad')
        )
        if option == 'Simulación de Monte Carlo':
            st.subheader("Simulación de Monte Carlo")
            n = st.session_state.N
            df_retornos = st.session_state.ret_log
            df_portafolio = montecarlo(1000, n, df_retornos)
            st.markdown('Dataframe de simulaciones con sus retornos, volatilidad, Sharpe y pesos por orden de activo.')
            st.write(df_portafolio)

            max_sh = df_portafolio.iloc[df_portafolio.Sharpe.idxmax()]
            min_vol = df_portafolio.iloc[df_portafolio.Volatibilidad.idxmin()]
            max_ret = df_portafolio.iloc[df_portafolio.Retornos.idxmax()]
            fig = px.scatter(
                df_portafolio, x='Volatibilidad', y='Retornos', color='Sharpe',
                labels={'Volatibilidad': 'Volatilidad esperada', 'Retornos': 'Retornos esperados'},
                title='Optimización de Portfolio'
            )

            fig.add_scatter(
                x=[min_vol[1]], y=[min_vol[0]], mode='markers',
                marker=dict(color='black', symbol='star', size=20), name='Máxima volatilidad esperada'
            )
            fig.add_scatter(
                x=[max_ret[1]], y=[max_ret[0]], mode='markers',
                marker=dict(color='black', symbol='star', size=20), name='Máximos rendimientos esperados'
            )
            fig.add_scatter(
                x=[max_sh[1]], y=[max_sh[0]], mode='markers',
                marker=dict(color='black', symbol='star', size=20), name='Máximo Sharpe Ratio'
            )

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

            st.write('-' * 100)
            st.text('Mejor Sharpe ratio:')
            st.write(max_sh)
            st.write('-' * 100)
            st.text("Menor volatilidad:")
            st.write(min_vol)
            st.write('-' * 100)

            st.text("Mayor rendimiento:")
            st.write(max_ret)
            st.write('-' * 100)

            df = st.session_state.df
            sp = st.session_state.Sp
            st.subheader("Benchmark")
            lista_pesos = st.selectbox('Seleccione los pesos para cada activo:', df_portafolio.Pesos)
            bench = benchmkark(df, sp, lista_pesos)
            st.write(bench)
            st.subheader('Benchmark, Valor del portafolio vs SP500')
            st.line_chart(bench[['Value', 'SP500']])

        elif option == 'Optimización por Sharpe ratio':
            n = st.session_state.N
            df_retornos = st.session_state.ret_log

            st.subheader('Optimización del portafolio sobre el Sharpe ratio')
            bounds = tuple((0, 1) for symbol in range(n))
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            init_guess = n * [1 / n]

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
                'Pesos': [optimized_sharpe.x]
            })
            st.write(result)

            df = st.session_state.df
            sp = st.session_state.Sp

            st.subheader("Benchmark")
            lista_pesos = st.selectbox('Seleccione los pesos para cada activo:', result.Pesos)
            bench = benchmkark(df, sp, lista_pesos)
            st.write(bench)
            st.subheader('Benchmark, Valor del portafolio vs SP500')
            st.line_chart(bench[['Value', 'SP500']])

        elif option == 'Optimización por volatilidad':
            n = st.session_state.N
            df_retornos = st.session_state.ret_log

            st.subheader('Optimización del portafolio sobre la volatilidad')
            bounds = tuple((0, 1) for symbol in range(n))
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            init_guess = n * [1 / n]

            optimized_vol = sci_opt.minimize(
                get_vol,
                init_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            optimized_metrics = calculos_(weights=optimized_vol.x)
            st.write('Resultados:')
            result = pd.DataFrame({
                'Retornos': optimized_metrics[0],
                'Volatilidad': optimized_metrics[1],
                'Sharpe': optimized_metrics[2],
                'Pesos': [optimized_vol.x]
            })
            st.write(result)

            df = st.session_state.df
            sp = st.session_state.Sp

            st.subheader("Benchmark")
            lista_pesos = st.selectbox('Seleccione los pesos para cada activo:', result.Pesos)
            bench = benchmkark(df, sp, lista_pesos)
            st.write(bench)
            st.subheader('Benchmark, Valor del portafolio vs SP500')
            st.line_chart(bench[['Value', 'SP500']])
else:
    st.text('Por favor, ingrese un número válido de activos mayores a 1.')
