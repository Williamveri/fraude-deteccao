from fastapi import FastAPI, Header, HTTPException, Depends
import joblib
import pandas as pd
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def gerar_explicacao(dados_transacao: dict, probabilidade: float, top_features: list):
    valores_relevantes = {f: round(dados_transacao[f], 3) for f in top_features}
    
    prompt = f"""Você é um assistente de análise de fraude bancária.
Uma transação foi sinalizada com probabilidade de fraude de {probabilidade:.1%}.

Os valores das variáveis mais relevantes para o modelo nessa transação foram:
{valores_relevantes}

Nota: as variáveis V1-V28 são anônimas (derivadas de PCA), então não descreva o que elas 
"significam" no mundo real. Em vez disso, explique de forma breve e objetiva (2-3 frases, 
em português) por que o padrão numérico observado é atípico, focando em conceitos gerais 
como magnitude e desvio do padrão comum, sem inventar significados de negócio para as variáveis."""

    resposta = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=150)
    )
    
    return resposta.text

app = FastAPI(title="API de Detecção de Fraude")

modelo = joblib.load('modelo_fraude_xgb.pkl')
scaler = joblib.load('scaler.pkl')

with open('threshold.txt', 'r') as f:
    threshold = float(f.read())

from pydantic import BaseModel

class Transacao(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

top_features = ['V14', 'V4', 'V12', 'V8', 'Amount']  


API_KEY_ESPERADA = os.environ["MINHA_API_KEY"]
def verificar_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY_ESPERADA:
        raise HTTPException(status_code=401, detail="API key inválida")
    
@app.post("/score")
def prever_fraude(transacao: Transacao, autorizado: None = Depends(verificar_api_key)):
    dados_dict = transacao.model_dump()
    dados = pd.DataFrame([dados_dict])
    dados_escalados = scaler.transform(dados)
    
    probabilidade = modelo.predict_proba(dados_escalados)[0][1]
    decisao = "fraude" if probabilidade >= threshold else "normal"
    
    explicacao = None
    if decisao == "fraude":
        try:
            explicacao = gerar_explicacao(dados_dict, probabilidade, top_features)
        except Exception as e:
            explicacao = "Não foi possível gerar explicação no momento."
    
    return {
        "probabilidade": round(float(probabilidade), 4),
        "decisao": decisao,
        "threshold_usado": threshold,
        "explicacao": explicacao
    }