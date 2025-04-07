import os
import pandas as pd

# Caminho da pasta onde os arquivos de saída estão localizados
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Lista dos arquivos de saída
arquivos_saida = ['transformado_lp1.csv', 'transformado_lp2.csv', 'transformado_lp3.csv', 
                  'transformado_lp4.csv', 'transformado_lp5.csv']

# Função para extrair e mostrar os valores únicos de 'label' em cada arquivo de saída
def obter_labels_unicos_arquivos_saida(file_path):
    try:
        # Ler o arquivo CSV usando pandas
        df = pd.read_csv(file_path)

        # Mostrar as colunas do arquivo para ajudar na depuração
        print(f"Colunas do arquivo {file_path}: {df.columns}")

        # Verificar se a coluna 'label' existe
        if 'label' not in df.columns:
            print(f"Erro: O arquivo {file_path} não contém a coluna 'label'.")
            return None

        # Usar o método unique para pegar os valores únicos da coluna 'label'
        return df['label'].unique()

    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None

# Função principal para processar todos os arquivos de saída e mostrar os labels únicos
def exibir_labels_unicos_arquivos_saida():
    for arquivo in arquivos_saida:
        # Caminho completo do arquivo de saída
        file_path = os.path.join(caminho_da_pasta_saida, arquivo)

        # Obter os labels únicos do arquivo de saída
        labels_unicos = obter_labels_unicos_arquivos_saida(file_path)

        if labels_unicos is not None:
            print(f"Valores únicos de 'label' no arquivo {arquivo}: {', '.join(labels_unicos)}")
        else:
            print(f"Erro ao processar o arquivo {arquivo}")

# Função para verificar se as variáveis preditivas (Fx, Fy, ..., Tz) são iguais entre os datasets lp2 e lp3
def comparar_variaveis_preditoras(lp2_path, lp3_path):
    try:
        # Carregar os datasets lp2 e lp3
        df_lp2 = pd.read_csv(lp2_path)
        df_lp3 = pd.read_csv(lp3_path)

        # Comparar as colunas Fx, Fy, Fz, Tx, Ty, Tz entre os dois datasets
        variaveis_preditoras = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        
        for var in variaveis_preditoras:
            # Verificar se os valores são iguais entre os datasets
            if var in df_lp2.columns and var in df_lp3.columns:
                iguais = df_lp2[var].equals(df_lp3[var])
                if iguais:
                    print(f"Os valores da variável '{var}' são os mesmos em lp2 e lp3.")
                else:
                    print(f"Os valores da variável '{var}' são diferentes entre lp2 e lp3.")
            else:
                print(f"A variável '{var}' não está presente em ambos os datasets.")
                
    except Exception as e:
        print(f"Erro ao comparar os datasets lp2 e lp3: {e}")

# Rodar a função principal para exibir os labels únicos
exibir_labels_unicos_arquivos_saida()

# Comparar as variáveis preditivas (Fx, Fy, ..., Tz) nos datasets lp2 e lp3
lp2_path = os.path.join(caminho_da_pasta_saida, 'transformado_lp2.csv')
lp3_path = os.path.join(caminho_da_pasta_saida, 'transformado_lp3.csv')

comparar_variaveis_preditoras(lp2_path, lp3_path)
