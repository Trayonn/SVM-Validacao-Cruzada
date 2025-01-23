## SVM para Classificação de Estilo de Jogo de Futebol
Este projeto utiliza o algoritmo Support Vector Machine (SVM) para classificar jogadores de futebol com base em suas estatísticas de desempenho. O modelo é treinado para prever o estilo de jogo de um jogador (Ataque ou Defesa) com base em dados como gols, assistências, dribles completos, interceptações, passes e distância percorrida.

## Descrição do Código
O código realiza as seguintes etapas:

- Criação e Expansão de Dados:

    O código começa com um conjunto de dados inicial contendo informações sobre seis jogadores e suas estatísticas de desempenho.
    Para simular 30 dias de treinamento, os dados são expandidos com variações aleatórias nas estatísticas dos jogadores. Isso cria um conjunto de dados mais robusto para treinar o modelo.
  
- Pré-processamento dos Dados:

    O DataFrame expandido é dividido em variáveis independentes (X) e rótulos (y).
    As variáveis independentes representam as estatísticas dos jogadores, enquanto os rótulos indicam o estilo de jogo (Ataque ou Defesa).
    Os dados são normalizados usando StandardScaler para garantir que todas as variáveis tenham a mesma escala, o que é importante para o desempenho do modelo SVM.
  
- Divisão dos Dados:

    O conjunto de dados é dividido em dois grupos: treino e teste. Aproximadamente 33% dos dados são usados para testar a precisão do modelo, enquanto o restante é utilizado para treinar o modelo.

- Treinamento do Modelo SVM:

    O modelo SVM é treinado com o kernel linear e o parâmetro de regularização C=1 usando os dados de treino.
    O modelo é avaliado utilizando os dados de teste, e as previsões feitas pelo modelo são comparadas com os rótulos reais.
  
- Geração de Relatório de Classificação:

    O código gera um relatório de classificação que avalia o desempenho do modelo SVM, apresentando métricas como:
    Precisão (Precision): Proporção de previsões corretas positivas entre todas as previsões positivas feitas.
    Revocação (Recall): Proporção de verdadeiros positivos entre todas as instâncias positivas.
    F1-Score: Média harmônica entre a precisão e a revocação.

## Dependências

Este projeto requer as seguintes bibliotecas Python:

- pandas: Para manipulação e análise de dados.
- numpy: Para cálculos e operações matemáticas.
- scikit-learn: Para as ferramentas de aprendizado de máquina, como o SVM e a normalização dos dados.
