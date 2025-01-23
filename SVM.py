import pandas as pd #é usado para manipulação de dados.
import numpy as np #é usada para cálculos
from sklearn.preprocessing import StandardScaler #Normalizar os dados, buscando precisão
from sklearn.svm import SVC #Usado para o SVM, será treinado para a classificação dos jgoadores
from sklearn.model_selection import train_test_split, cross_val_score #Dividir os dados em conjuntos de treino e teste
from sklearn.metrics import classification_report #Gera um relatório de métricas de avaliação (precisão, recall e F1-score)

# Dados iniciais dos jogadores
data = {
    'Jogador': ['João Silva', 'Pedro Santos', 'Lucas Ferreira', 'Carlos Lima', 'Rafael Souza', 'Marcos Reis'],
    'Gols': [25, 30, 20, 5, 3, 4],
    'Assistências': [15, 18, 12, 3, 2, 4],
    'Dribles Completos': [50, 45, 35, 12, 10, 8],
    'Interceptações': [5, 3, 4, 28, 32, 25],
    'Passes': [500, 550, 580, 950, 900, 870],
    'Distância Percorrida': [9.0, 8.5, 9.8, 11.5, 12.0, 11.2],
    'Estilo': ['Ataque', 'Ataque', 'Ataque', 'Defesa', 'Defesa', 'Defesa']
}

# Criar DataFrame
df = pd.DataFrame(data)

# Aumentar a base de dados para simular "dias de treinamento"
expanded_data = []

for i in range(30):  # 30 dias de treinamento para cada jogador
    for idx, row in df.iterrows():
        jogador_data = row.copy()
        jogador_data['Gols'] = max(0, row['Gols'] + np.random.randint(-2, 3))  # Variação de -2 a +2 gols
        jogador_data['Assistências'] = max(0, row['Assistências'] + np.random.randint(-2, 3))  # Variação de -2 a +2 assistências
        jogador_data['Dribles Completos'] = max(0, row['Dribles Completos'] + np.random.randint(-5, 6))  # Variação de -5 a +5 dribles
        jogador_data['Interceptações'] = max(0, row['Interceptações'] + np.random.randint(-3, 4))  # Variação de -3 a +3 interceptações
        jogador_data['Passes'] = max(0, row['Passes'] + np.random.randint(-50, 51))  # Variação de -50 a +50 passes
        jogador_data['Distância Percorrida'] = round(max(0, row['Distância Percorrida'] + np.random.uniform(-0.5, 0.5)), 2)  # Variação de -0.5 a +0.5 km
        expanded_data.append(jogador_data)

expanded_df = pd.DataFrame(expanded_data)

# Separar os recursos e o rótulo
X = expanded_df[['Gols', 'Assistências', 'Dribles Completos', 'Interceptações', 'Passes', 'Distância Percorrida']]
y = expanded_df['Estilo']

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SVM com validação cruzada
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))
