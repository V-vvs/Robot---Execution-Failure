import os
import pandas as pd

# Caminho da pasta onde os arquivos de saída estão localizados
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Lista dos arquivos de entrada
arquivos_entrada = ['transformado_lp1.csv', 'transformado_lp2.csv', 'transformado_lp4.csv']

# Função para transformar o label conforme as regras
def transformar_label(df, arquivo_nome):
    if arquivo_nome == 'transformado_lp1.csv':
        # Regras para lp1
        df['label'] = df['label'].replace({
            'normal': 'normal',
            'collision': 'collision',
            'obstruction': 'obstruction',
            'fr_collision': 'collision'
        })
    elif arquivo_nome == 'transformado_lp2.csv':
        # Regras para lp2
        df['label'] = df['label'].replace({
            'normal': 'normal',
            'back_col': 'collision',
            'front_col': 'collision',
            'right_col': 'collision',
            'left_col': 'collision'
        })
    elif arquivo_nome == 'transformado_lp4.csv':
        # Regras para lp4
        df['label'] = df['label'].replace({
            'normal': 'normal',
            'collision': 'collision',
            'obstruction': 'obstruction'
        })
    return df

# Função para carregar e processar os arquivos de entrada
def carregar_e_transformar_arquivos():
    amostras = []
    
    for arquivo in arquivos_entrada:
        # Caminho completo do arquivo de entrada
        file_path = os.path.join(caminho_da_pasta_saida, arquivo)
        
        # Carregar o arquivo CSV
        df = pd.read_csv(file_path)
        
        # Transformar os labels conforme as regras
        df = transformar_label(df, arquivo)
        
        # Adicionar as amostras transformadas ao conjunto final
        amostras.append(df)
    
    # Concatenar todos os datasets em um único
    df_final = pd.concat(amostras, ignore_index=True)
    
    return df_final

# Função para salvar o arquivo final
def salvar_arquivo_final(df_final):
    # Caminho do arquivo de saída
    caminho_saida = os.path.join(caminho_da_pasta_saida, 'dataset_robot.csv')
    
    # Salvar o dataset final em um arquivo CSV
    df_final.to_csv(caminho_saida, index=False)
    print(f"Arquivo final salvo como {caminho_saida}")

# Função para imprimir a quantidade de amostras por classe
def imprimir_quantidade_amostras_por_classe(df_final):
    # Contar a quantidade de amostras para cada classe
    contagem_classes = df_final['label'].value_counts()
    print("\nQuantidade de amostras para cada classe:")
    print(contagem_classes)

# Rodar o processo de transformação e salvar o arquivo final
df_final = carregar_e_transformar_arquivos()

# Salvar o arquivo final
salvar_arquivo_final(df_final)

# Imprimir a quantidade de amostras por classe
imprimir_quantidade_amostras_por_classe(df_final)
