import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Painel de Detecção de Fraude", layout="wide")
st.title("🔍 Painel de Detecção de Fraude em Tempo Real")

API_URL = "https://fraude-deteccao.onrender.com/score"
API_KEY = "7396d0b180c18e091e9d664630a432e6"

df = pd.read_csv('amostra_transacoes.csv')

if 'historico' not in st.session_state:
    st.session_state.historico = []

if st.button("▶️ Iniciar simulação"):
    placeholder = st.empty()
    
    for i, linha in df.iterrows():
        dados = linha.to_dict()
        
        resposta = requests.post(
            API_URL,
            json=dados,
            headers={"x-api-key": API_KEY}
        )
        resultado = resposta.json()
        
        registro = {
            "Transação": i,
            "Valor (R$)": round(dados['Amount'], 2),
            "Probabilidade": resultado['probabilidade'],
            "Decisão": resultado['decisao'],
            "Explicação": resultado['explicacao'] or '-'
        }
        st.session_state.historico.append(registro)
        
        with placeholder.container():
            historico_df = pd.DataFrame(st.session_state.historico)
            st.dataframe(historico_df, use_container_width=True)
        
        time.sleep(1.5)