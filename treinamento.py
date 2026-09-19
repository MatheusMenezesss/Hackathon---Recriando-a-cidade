"""
=====================================================================
MOTOR PREDITIVO DE ZELADORIA E ORDEM PÚBLICA - TREINAMENTO DO MODELO
=====================================================================
Entrada: dataset_seops.csv (gerado por coleta_dados_seops.py)
Saída:   motor_preditivo_seops.pkl   -> modelo treinado
         features_modelo.pkl        -> lista exata das colunas de X,
                                        na ordem certa, para a API
                                        montar o mesmo vetor na inferência
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

ARQUIVO_DATASET = "dataset_seops.csv"
ARQUIVO_MODELO = "motor_preditivo_seops.pkl"
ARQUIVO_FEATURES = "features_modelo.pkl"

# =====================================================================
# 0. CARGA DA BASE JÁ ESTRUTURADA
# =====================================================================
print("0. Carregando dataset estruturado...")
df = pd.read_csv(ARQUIVO_DATASET)
print(f"   {len(df)} registros carregados. Colunas: {df.columns.tolist()}")

# =====================================================================
# 1. TRATAMENTO DE CATEGÓRICAS (ENCODING)
# =====================================================================
print("1. Aplicando One-Hot Encoding em 'local' e 'iluminacao_fonte'...")
df_ml = pd.get_dummies(df, columns=['local', 'iluminacao_fonte'], drop_first=True)

# =====================================================================
# 2. SEPARAÇÃO DE FEATURES E TARGET (X e y)
# =====================================================================
print("2. Separando X (features) e y (target)...")
X = df_ml.drop('acao_preventiva_necessaria', axis=1)
y = df_ml['acao_preventiva_necessaria']
print(f"   X: {X.shape[1]} features | y: distribuição de classes -> "
      f"{y.value_counts(normalize=True).round(3).to_dict()}")

# =====================================================================
# 3. DIVISÃO ESTRATIFICADA (TRAIN / TEST SPLIT)
# =====================================================================
print("3. Dividindo treino/teste (80/20, estratificado por y)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 4. TREINAMENTO
# =====================================================================
print("4. Treinando RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
modelo.fit(X_train, y_train)

# =====================================================================
# 5. VALIDAÇÃO (métricas para o pitch)
# =====================================================================
print("5. Validando modelo no conjunto de teste...")
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print(f"\n--- MÉTRICAS DO MODELO ---")
print(f"Acurácia: {acuracia * 100:.2f}%")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

importancias = sorted(
    zip(X.columns, modelo.feature_importances_), key=lambda t: t[1], reverse=True
)
print("--- TOP 10 VARIÁVEIS MAIS IMPORTANTES (para a equipe de Negócios) ---")
for nome, imp in importancias[:10]:
    print(f"{nome}: {imp * 100:.1f}%")

# =====================================================================
# 6. SALVAMENTO DOS METADADOS (passo crítico para a API)
# =====================================================================
print("\n6. Salvando modelo e lista de features...")
joblib.dump(modelo, ARQUIVO_MODELO)
joblib.dump(list(X.columns), ARQUIVO_FEATURES)

print(f"✅ SUCESSO! '{ARQUIVO_MODELO}' e '{ARQUIVO_FEATURES}' salvos.")
print("   Na API: carregue features_modelo.pkl e use df_novo.reindex(columns=features, fill_value=0)")
print("   antes do modelo.predict(), pra garantir o mesmo alinhamento de colunas do treino.")