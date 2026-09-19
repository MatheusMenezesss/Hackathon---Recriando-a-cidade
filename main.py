import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Carregamento dos artefatos
modelo = joblib.load("motor_preditivo_seops.pkl")
features_modelo = joblib.load("features_modelo.pkl")

app = FastAPI(
    title="SEOPS Recife - Motor Preditivo de Zeladoria e Ordem Pública",
    version="1.0.0"
)

# Habilita CORS para o Frontend/Figma/Postman consumir sem bloqueio de rede
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Schema de Entrada (exatamente as colunas brutas antes do get_dummies)
class EntradaPredicao(BaseModel):
    local: str
    dia_semana: int               # 0 a 6
    hora_dia: int                 # 0 a 23
    eventos_proximos: int         # 0 ou 1
    historico_ocorrencias_7d: int
    iluminacao_ativa_pct: int     # 0 a 100
    iluminacao_fonte: str
    densidade_pessoas: int        # 0 a 100

@app.post("/api/v1/predicao-risco")
def prever_risco(dados: EntradaPredicao):
    # Converte payload em DataFrame de 1 linha
    df_entrada = pd.DataFrame([dados.model_dump()])

    # Aplica o mesmo get_dummies do treino
    df_ml = pd.get_dummies(df_entrada, columns=['local', 'iluminacao_fonte'], drop_first=True)

    # Garante alinhamento exato: colunas faltantes viram 0, ordem preservada
    df_alinhado = df_ml.reindex(columns=features_modelo, fill_value=0)

    # Inferência
    predicao = int(modelo.predict(df_alinhado)[0])
    probabilidade = float(modelo.predict_proba(df_alinhado)[0][1])

    return {
        "local": dados.local,
        "acao_preventiva_necessaria": predicao == 1,
        "probabilidade_risco": round(probabilidade, 3),
        "nivel_criticidade": "ALTO" if probabilidade >= 0.6 else "MEDIO" if probabilidade >= 0.3 else "BAIXO",
        "recomendacao_operacional": (
            "Despachar viatura e equipe preventiva de conservação"
            if predicao == 1
            else "Manter ronda ordinária"
        )
    }

@app.get("/api/v1/health")
def health_check():
    return {"status": "operacional", "modelo_carregado": True}