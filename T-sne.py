import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler  # Usando MinMaxScaler
from sklearn.manifold import TSNE
import ast

# Caminho do arquivo de saída (onde está o arquivo final)
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Caminho do arquivo de entrada (o arquivo final gerado)
arquivo_entrada = 'dataset_robot.csv'

# Função para carregar o arquivo e processar os dados
def carregar_e_processar_dados():
    # Carregar o arquivo de saída
    file_path = os.path.join(caminho_da_pasta_saida, arquivo_entrada)
    df = pd.read_csv(file_path)

    # Função para garantir que as variáveis Fx, Fy, Fz, etc., sejam listas numéricas
    def converter_para_lista(texto):
        try:
            # Tenta converter a string para uma lista de números
            return ast.literal_eval(texto)
        except:
            # Caso falhe, retorna um valor numérico padrão (0.0)
            return [0.0]

    # Aplicar a conversão para todas as variáveis que são listas em formato de texto
    df['Fx'] = df['Fx'].apply(converter_para_lista)
    df['Fy'] = df['Fy'].apply(converter_para_lista)
    df['Fz'] = df['Fz'].apply(converter_para_lista)
    df['Tx'] = df['Tx'].apply(converter_para_lista)
    df['Ty'] = df['Ty'].apply(converter_para_lista)
    df['Tz'] = df['Tz'].apply(converter_para_lista)

    return df

# Função para transformar as listas em colunas separadas e normalizar os dados
def normalizar_variavel(df, variavel):
    # Transformar a lista de cada linha para um array de valores
    variavel_values = pd.DataFrame(df[variavel].tolist(), index=df.index)

    # Normalizar os dados usando Min-Max (para o intervalo [-1, 1])
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Aqui é o intervalo de -1 a 1
    variavel_normalizado = scaler.fit_transform(variavel_values)

    return variavel_normalizado, df['label']

# Função para aplicar t-SNE e plotar os resultados
def plotar_tsne(variavel_normalizado, labels, variavel):
    # Configuração do t-SNE com perplexidade = 30
    tsne = TSNE(n_components=2, random_state=42, perplexity=200)

    # Aplicar t-SNE para reduzir a dimensionalidade para 2 componentes
    X_tsne = tsne.fit_transform(variavel_normalizado)

    # Criar o DataFrame para visualização
    tsne_df = pd.DataFrame(X_tsne, columns=['Dim 1', 'Dim 2'])
    tsne_df['label'] = labels

    # Paleta de cores personalizada (azul para normal, amarelo para collision e laranja para obstruction)
    paleta = {'normal': 'blue', 'collision': 'yellow', 'obstruction': 'orange'}

    # Plotar
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=tsne_df, x='Dim 1', y='Dim 2', hue='label', palette=paleta, s=80, alpha=0.7)
    plt.title(f't-SNE: Comparando {variavel}')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend(title='Classe', loc='best')
    plt.grid(True)
    plt.show()

# Carregar e processar os dados
df = carregar_e_processar_dados()

# Gerar o t-SNE para cada variável individualmente
variaveis = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

for variavel in variaveis:
    variavel_normalizado, labels = normalizar_variavel(df, variavel)
    plotar_tsne(variavel_normalizado, labels, variavel)
