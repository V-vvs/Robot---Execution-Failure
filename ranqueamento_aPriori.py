import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from skrebate import ReliefF
import ast

# Caminho do arquivo de entrada
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'
arquivo_entrada = 'dataset_robot.csv'

# Função para carregar e processar os dados
def carregar_e_processar_dados():
    file_path = os.path.join(caminho_da_pasta_saida, arquivo_entrada)
    df = pd.read_csv(file_path)

    # Função para garantir que as variáveis Fx, Fy, Fz, etc., sejam listas numéricas
    def converter_para_lista(texto):
        try:
            return ast.literal_eval(texto)
        except:
            return [0.0]

    # Aplicar a conversão para todas as variáveis que são listas em formato de texto
    for col in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
        df[col] = df[col].apply(converter_para_lista)

    # Converter as listas em valores de resumo (média) para cada variável
    for col in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
        df[col] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)

    # As variáveis de interesse para o ranqueamento
    X = df[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']]
    y = df['label']
    
    return X, y

# Função para aplicar o método Chi-Square
def chi2_ranking(X, y):
    # Ajuste necessário: Chi-Square só funciona com variáveis numéricas não-negativas.
    # Aqui aplicamos uma transformação para garantir que os valores sejam não-negativos.
    X_chi2 = X.copy()
    X_chi2 = X_chi2.applymap(lambda x: max(0, x))  # Garantir que os valores sejam não-negativos
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_chi2, y)
    chi2_scores = chi2_selector.scores_
    return chi2_scores

# Função para aplicar o método ReliefF
def reliefF_ranking(X, y):
    relief = ReliefF(n_neighbors=10)
    relief.fit(X.to_numpy(), y)
    return relief.feature_importances_

# Função para aplicar o método F-Test
def f_test_ranking(X, y):
    f_selector = SelectKBest(f_classif, k='all')
    f_selector.fit(X, y)
    f_scores = f_selector.scores_
    return f_scores

# Carregar e processar os dados
X, y = carregar_e_processar_dados()

# Aplicar os 3 métodos de ranqueamento
chi2_scores = chi2_ranking(X, y)
relief_scores = reliefF_ranking(X, y)
f_test_scores = f_test_ranking(X, y)

# Criar um DataFrame para visualizar os rankings
ranking_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'ReliefF Score': relief_scores,
    'F-Test Score': f_test_scores
})

# Ordenar as variáveis de acordo com os escores (do maior para o menor)
ranking_df['Chi2 Rank'] = ranking_df['Chi2 Score'].rank(ascending=False)
ranking_df['ReliefF Rank'] = ranking_df['ReliefF Score'].rank(ascending=False)
ranking_df['F-Test Rank'] = ranking_df['F-Test Score'].rank(ascending=False)

# Imprimir os scores de ranqueamento
print("Scores de Ranqueamento - Chi-Square:")
print(ranking_df[['Feature', 'Chi2 Score']])
print("\nScores de Ranqueamento - ReliefF:")
print(ranking_df[['Feature', 'ReliefF Score']])
print("\nScores de Ranqueamento - F-Test:")
print(ranking_df[['Feature', 'F-Test Score']])

# Plotando os rankings comparativos
plt.figure(figsize=(18, 10))

# Gráfico para Chi-Square (ordenado pelos escores)
plt.subplot(1, 3, 1)  # Subgráfico 1 (1 linha, 3 colunas, posição 1)
ranking_df.sort_values(by='Chi2 Score', ascending=False).set_index('Feature')[['Chi2 Score']].plot(kind='barh', ax=plt.gca(), color='skyblue')
plt.title('Ranking das Features usando Chi-Square')
plt.xlabel('Chi2 Score')
plt.ylabel('Feature')
plt.grid(True)

# Gráfico para ReliefF (ordenado pelos escores)
plt.subplot(1, 3, 2)  # Subgráfico 2 (1 linha, 3 colunas, posição 2)
ranking_df.sort_values(by='ReliefF Score', ascending=False).set_index('Feature')[['ReliefF Score']].plot(kind='barh', ax=plt.gca(), color='lightgreen')
plt.title('Ranking das Features usando ReliefF')
plt.xlabel('ReliefF Score')
plt.ylabel('Feature')
plt.grid(True)

# Gráfico para F-Test (ordenado pelos escores)
plt.subplot(1, 3, 3)  # Subgráfico 3 (1 linha, 3 colunas, posição 3)
ranking_df.sort_values(by='F-Test Score', ascending=False).set_index('Feature')[['F-Test Score']].plot(kind='barh', ax=plt.gca(), color='lightcoral')
plt.title('Ranking das Features usando F-Test')
plt.xlabel('F-Test Score')
plt.ylabel('Feature')
plt.grid(True)

# Ajustando layout para evitar sobreposição
plt.tight_layout()

# Salvar a figura como um arquivo PNG
output_path = os.path.join(caminho_da_pasta_saida, 'ranking_features.png')
plt.savefig(output_path)

# Exibir a figura
plt.show()

# Exibindo o DataFrame com os rankings
print(ranking_df)
