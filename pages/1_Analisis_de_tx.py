import time
import pandas as pd
import matplotlib.pyplot as plt
from requests import get
from datetime import datetime
import streamlit as st

st.set_page_config(
    page_title = "Analisis de Eth wallets.",
    page_icon = "🤖",
)

API_KEY='NZXSEDDABZ6ZTB5RMRUIBDQPVE6KS6W2SY'
BASE_URL = "https://api.etherscan.io/api"
ETHER_VALUE = 10**18

def make_api_url(module, action, address, **kwargs):
	url = BASE_URL + f"?module={module}&action={action}&address={address}&apikey={API_KEY}"

	for key, value in kwargs.items():
		url += f"&{key}={value}"

	return url

def get_account_balance(address):
	balance_url = make_api_url("account", "balance", address, tag="latest")
	response = get(balance_url)
	data = response.json()

	value = int(data["result"]) / ETHER_VALUE
	return value

def get_transactions(address):
  transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
  response = get(transactions_url)
  data = response.json()["result"]
  df = pd.DataFrame(data)
  df = df.filter(['blockNumber', 'timeStamp', 'from', 'to', 'value', 'gasPrice', 'gasUsed'], axis=1)
  df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
  columnas_ = ['blockNumber', 'value', 'gasPrice', 'gasUsed']
  for i in columnas_:
    df[i] = pd.to_numeric(df[i], errors='coerce')

  df['value(Eth)'] = df.value / ETHER_VALUE
  df['Gas(Eth)'] = (df.gasPrice * df.gasUsed) / ETHER_VALUE
  return df.set_index('blockNumber').drop(['value'], axis=1)

def get_tokens_tx(address):
  get_tokens_tx_url = make_api_url('account', 'tokentx', address, page=1, offset=150, startblock = 0, endblock = 27025780, sort = 'dsc')

  response = get(get_tokens_tx_url)
  if response.json()['message'] == 'No transactions found':
    st.write("'No existen tx con este tipo de tokens'")
  else:
      data = pd.DataFrame(response.json()["result"])
      df = data.filter(['blockNumber',	'timeStamp','from', 'to', 'contractAddress', 'value', 'tokenName', 'tokenSymbol', 'tokenDecimal'], axis=1)
      df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
      columnas_ = ['blockNumber', 'value', 'tokenDecimal']
      for i in columnas_:
        df[i] = pd.to_numeric(df[i], errors='coerce')
      df.value = df.value  / (10 ** df.tokenDecimal)
      return df.set_index('blockNumber')

def get_nft_response(address):
    get_tokens_tx_url = make_api_url('account', 'tokennfttx', address, page = 1, offset=150, startblock = 0, endblock = 27025780, sort = 'dsc')
    resp = get(get_tokens_tx_url)
    return resp.json()

def edit_nft(response):
    data = pd.DataFrame(response)
    df = data.filter(['blockNumber', 'timeStamp', 'from', 'to', 'contractAddress', 'tokenName', 'tokenSymbol', 'tokenDecimal'], axis=1)
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    columnas_ = ['blockNumber', 'tokenDecimal']
    for i in columnas_:
      df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.set_index('blockNumber')
    return df

def get_balance_erc20(address, contractaddress, decimales):
    get_erc20_balance = make_api_url('account', 'tokenbalance', address, contractaddress = contractaddress )
    response = get(get_erc20_balance)
    data = response.json()["result"]
    return float(data) / (10**decimales)

def get_eth_px():
  px_url = BASE_URL + f"?module={'stats'}&action={'ethprice'}&apikey={API_KEY}"
  response = get(px_url)
  rdo = response.json()["result"]
  return rdo['ethusd']

def get_balance_usd(address):
    eth_balance = get_account_balance(address)
    ethusd = round(float(get_eth_px()) * eth_balance, 2)
    
    return eth_balance, ethusd 

@st.cache_data 
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

st.title("Analisis de Ethereum addresses")
st.text('Address de prueba: 0x270963D9085E924Cbc98085859D0d9532aCD8d02')
address = (st.text_input("Ingrese una direccion de Ethereum:", key='Address')).lower()
if 'Address' not in st.session_state:
    st.write("Primero ingrese la direccion de Ethereum.")

else:
    option = st.selectbox(
        '¿Que te gustaria hacer?',
        (
        'Ingrese los datos primero.',
         'Conocer el balance de Ethereum de una direccion', 
         'Obtener sus tx', 
         'Obtener sus ERC20',
         'Obtener sus NFTS'
         ))
    
    if option == 'Conocer el balance de Ethereum de una direccion':
        st.subheader("Balance: ")
        eth_balance, ethusd  = get_balance_usd(address)
        st.text(f"El usuario tiene {eth_balance}ETH en su wallet.")
        st.text(f"Expresado en usd: ${ethusd}USD. Precio actual: ${get_eth_px()}")
    elif option == 'Obtener sus tx':
        data = get_transactions(address)
        st.subheader('Tx Dataframe:')
        st.write(data)
        csv_tx = convert_df(data)
        st.download_button("▼ Descargar.", csv_tx, "tx_data.csv", "text/csv",key='download-csv')
        
        st.subheader('Estadisticas.')
        st.write("-" * 100)
        recib = data.loc[data['from'] != address]
        env = data.loc[data['from'] == address]
    
        total_recib, total_env = len(recib['from']), len(env['from'])
        st.text(f"Cantidad de tx recibidas: {total_recib}")
        st.text(f"Cantidad de tx enviadas: {total_env}")
        st.write("-" * 100)
        
        total_eth_recib, total_eth_env = recib['value(Eth)'].sum(), env['value(Eth)'].sum()
        st.text(f"Eth recibido: {total_eth_recib}")
        st.text(f"Eth enviado: {total_eth_env}")
        st.write("-" * 100)
        
        st.subheader('Grafico de barras sobre las direcciones que han enviado ETH.')
        x = recib['from'].value_counts()
        y = x.values
        
        fig, ax = plt.subplots()  
        ax.barh(range(len(x)), y)  # Usar range(len(x)) como posiciones de las barras en el eje y
        ax.set_yticks(range(len(x)))  # Establecer las ubicaciones de las etiquetas del eje y
        ax.set_yticklabels(x.index)  # Establecer las etiquetas del eje y como los índices de x        
        st.pyplot(fig)
        st.write("-" * 100)
        
        st.subheader('Grafico de barras sobre tx enviadas por la address.')
        x = env['to'].value_counts()
        y = x.values
        
        fig, ax = plt.subplots()  
        ax.barh(range(len(x)), y)  # Usar range(len(x)) como posiciones de las barras en el eje y
        ax.set_yticks(range(len(x)))  # Establecer las ubicaciones de las etiquetas del eje y
        ax.set_yticklabels(x.index)  # Establecer las etiquetas del eje y como los índices de x        
        st.pyplot(fig)
        
    elif option == 'Obtener sus ERC20':
        st.subheader('Tokens ERC20 transferidos/recibidos.')
        erc = get_tokens_tx(address)
        st.write(erc)
        csv_erc = convert_df(erc)
        st.download_button("▼ Descargar.", csv_erc, "erc_data.csv", "text/csv",key='download-csv')
        st.write("-" * 100)
        
        recib = erc.loc[erc['from'] != address]
        env = erc.loc[erc['from'] == address]
        st.subheader("Pie chart sobre los tokens ERC20 recibidos:")
        x = recib['from'].value_counts()
        
        if len(x) <= 0:
            st.text('Esta address no tiene tx recibidas.')
        else:
            df = recib.groupby(['tokenSymbol', 'from']).size().reset_index(name='from_count')
            fig, ax = plt.subplots()
            ax.pie(df['from_count'], labels=df['tokenSymbol'])
            st.pyplot(fig)
        st.write("-" * 100)    
        
        st.subheader("Pie chart sobre los tokens ERC20 envidados:")    
        x = env['to'].value_counts()
        if len(x) <= 0:
            st.text('Esta address no ha enviado tokens.')
        else:
            df = env.groupby(['tokenSymbol', 'to']).size().reset_index(name='from_to')
            fig, ax = plt.subplots()
            ax.pie(df['from_to'], labels = df['tokenSymbol'])
            st.pyplot(fig)
        st.write("-" * 100)    
        
        st.subheader("Balance por token.")
        st.text('ContractAddress: 0x55db959d60cd17d7d252524c6773f3f6de87846d')
        st.text('Decimales: 18')
        erc20 = (st.text_input('Ingrese un token ERC20 y obtenga el balance del usuario:')).lower()    
        decim = st.number_input('Ingrese los decimales del token:', min_value=0, max_value=21, key='decim')
        
        if 'decim' in st.session_state:
            balanc = get_balance_erc20(address, erc20, decim)
            st.text(f'El usuario tiene un balance de: {balanc} tokens.')
        else:
            st.write('Complete los datos!')
            
       
    elif option == 'Obtener sus NFTS':
        st.subheader("Dataframe sobre tokens ERC721.")
        resp = get_nft_response(address)
        if resp['message'] == 'No transactions found':
            st.text("'No existen tx con este tipo de tokens'")
                   
        else:
            data = edit_nft(resp["result"])
            st.write(data)
            data_csv = convert_df(data)
            st.download_button("▼ Descargar.", data_csv, "nft_data.csv", "text/csv",key='download-csv')
        st.write("-" * 100)    

        

    
