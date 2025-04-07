import os
import pandas as pd

# Caminho da pasta onde os arquivos .data estão localizados
caminho_da_pasta = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Lista dos arquivos .data
arquivos = ['lp1.data', 'lp2.data', 'lp3.data', 'lp4.data', 'lp5.data']

# Função para processar os dados
def processar_arquivo(file_path):
    try:
        # Listas para armazenar as amostras
        amostras = []
        fx, fy, fz, tx, ty, tz = [], [], [], [], [], []
        label = None  # Variável para armazenar o valor do target
        linhas_vazias = 0  # Contador de linhas vazias

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remover espaços extras

                # Se a linha for vazia, contar as linhas vazias
                if not line:
                    linhas_vazias += 1
                    # Se encontramos ao menos uma linha vazia, podemos tentar identificar o target
                    if linhas_vazias > 1 and label is not None:
                        # Se encontramos mais de uma linha vazia, assumimos que o target mudou
                        amostras.append({
                            'label': label,
                            'Fx': fx,
                            'Fy': fy,
                            'Fz': fz,
                            'Tx': tx,
                            'Ty': ty,
                            'Tz': tz
                        })
                        fx, fy, fz, tx, ty, tz = [], [], [], [], [], []  # Reset para a próxima amostra
                        linhas_vazias = 0  # Resetar contador de linhas vazias
                else:
                    linhas_vazias = 0  # Resetar contador de linhas vazias se a linha não for vazia
                    
                    # Se a linha for um target (um texto indicando o tipo, como 'normal', 'back_col', etc.)
                    if line.isalpha() or "_" in line:  # Agora também identificamos alvos com underscores
                        label = line.strip().lower()
                    else:
                        # Dividir a linha em valores e adicionar à lista apropriada
                        valores = line.split()
                        if len(valores) == 6:  # Verifica se há 6 valores (Fx, Fy, Fz, Tx, Ty, Tz)
                            fx.append(int(valores[0]))  # Fx
                            fy.append(int(valores[1]))  # Fy
                            fz.append(int(valores[2]))  # Fz
                            tx.append(int(valores[3]))  # Tx
                            ty.append(int(valores[4]))  # Ty
                            tz.append(int(valores[5]))  # Tz

            # Adicionar a última amostra (que não será adicionada dentro do loop)
            if len(fx) > 0 and label is not None:
                amostras.append({
                    'label': label,
                    'Fx': fx,
                    'Fy': fy,
                    'Fz': fz,
                    'Tx': tx,
                    'Ty': ty,
                    'Tz': tz
                })

        return amostras

    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None

# Função para salvar os dados transformados no formato desejado
def salvar_dados_transformados(amostras, nome_arquivo_saida):
    df = pd.DataFrame(amostras)
    # Salvar no formato CSV
    df.to_csv(nome_arquivo_saida, index=False)
    print(f"Arquivo transformado salvo como {nome_arquivo_saida}")

# Processar todos os arquivos
for arquivo in arquivos:
    # Caminho completo do arquivo de entrada
    file_path = os.path.join(caminho_da_pasta, arquivo)

    # Processa o arquivo e obtém as amostras
    amostras = processar_arquivo(file_path)

    if amostras is not None:
        # Nome do arquivo de saída com extensão .csv
        nome_arquivo_saida = os.path.join(caminho_da_pasta, f"transformado_{arquivo.replace('.data', '.csv')}")
        salvar_dados_transformados(amostras, nome_arquivo_saida)
