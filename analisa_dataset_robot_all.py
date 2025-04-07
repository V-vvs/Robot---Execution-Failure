import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Caminho do arquivo de saída (onde está o arquivo final)
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Caminho do arquivo de entrada (o arquivo final gerado)
arquivo_entrada = 'dataset_robot.csv'

# Caminho da pasta onde as imagens serão salvas
pasta_imagens = os.path.join(caminho_da_pasta_saida, 'perfis')

# Criar a pasta 'perfis' se não existir
if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)

# Função para carregar o arquivo e processar os dados
def carregar_e_processar_dados():
    # Carregar o arquivo de saída
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

    return df

# Função para normalizar as variáveis
def normalizar_variavel(df, variavel):
    # Normalizar as listas da variável utilizando MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Normalizar para o intervalo [-1, 1]
    variavel_normalizada = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[variavel].tolist())), index=df.index)
    return variavel_normalizada

# Função para calcular as médias das curvas para cada classe
def calcular_media_por_classe(df):
    # Definir as classes
    classes = ['normal', 'collision', 'obstruction']

    # Inicializar dicionário para armazenar as médias
    medias_classes = {classe: {var: [] for var in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']} for classe in classes}
    
    for classe in classes:
        # Filtrar dados por classe
        df_classe = df[df['label'] == classe]

        if df_classe.empty:
            print(f"Sem dados para a classe {classe}.")
            continue
        
        # Normalizar e calcular a média para cada variável
        for variavel in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
            variavel_normalizada = normalizar_variavel(df_classe, variavel)
            medias_classes[classe][variavel] = np.mean(variavel_normalizada.values, axis=0)

    return medias_classes

# Função para plotar as curvas médias das variáveis por classe
def plotar_curvas_medianas(medias_classes):
    # Para cada variável, plotar as curvas médias para cada classe
    for variavel in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
        plt.figure(figsize=(12, 8))
        for classe in ['normal', 'collision', 'obstruction']:
            plt.plot(medias_classes[classe][variavel], label=f'{classe} - {variavel}', lw=2)
        
        plt.title(f'Curva Média de {variavel} para cada Classe')
        plt.xlabel('Tempo (315ms por intervalo)')
        plt.ylabel('Valor Normalizado')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(pasta_imagens, f'curva_media_{variavel}.png'))
        plt.close()

# Carregar os dados e processar
df = carregar_e_processar_dados()

# Calcular as médias das curvas para cada classe
medias_classes = calcular_media_por_classe(df)

# Plotar as curvas médias
plotar_curvas_medianas(medias_classes)
