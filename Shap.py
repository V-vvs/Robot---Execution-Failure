import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Caminho do arquivo de saída (onde está o arquivo final)
caminho_da_pasta_saida = r'C:\Users\vanes\Downloads\robot+execution+failures\data'

# Caminho do arquivo de entrada (o arquivo final gerado)
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

    return df

# Função para normalizar as variáveis
def normalizar_variavel(df, variavel):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalizar para o intervalo [0, 1]
    variavel_normalizada = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[variavel].tolist())), index=df.index)
    return variavel_normalizada

# Função para treinar o modelo e calcular a importância das variáveis
def calcular_importancia_features(df):
    # Separar as variáveis de entrada (X) e as labels (y)
    X = np.column_stack([df['Fx'].apply(lambda x: np.mean(x)),
                         df['Fy'].apply(lambda x: np.mean(x)),
                         df['Fz'].apply(lambda x: np.mean(x)),
                         df['Tx'].apply(lambda x: np.mean(x)),
                         df['Ty'].apply(lambda x: np.mean(x)),
                         df['Tz'].apply(lambda x: np.mean(x))])

    # Labels (classes)
    y = df['label'].values

    # Garantir que as dimensões de X e y são consistentes
    if X.shape[0] != len(y):
        raise ValueError(f"Número de amostras de X ({X.shape[0]}) e y ({len(y)}) não coincide")

    # Dividir os dados em treino e teste (usando uma parte para validação do modelo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo de Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Importância das features
    importancias = rf.feature_importances_

    # Predições no conjunto de teste
    y_pred = rf.predict(X_test)

    # Relatório de classificação
    clf_report = classification_report(y_test, y_pred, target_names=rf.classes_)

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    return importancias, rf.classes_, clf_report, conf_matrix, rf, X_test, y_test, y_pred, X_train

# Função para plotar a importância das features
def plotar_importancia(importancias, classes):
    features = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    importancia_df = pd.DataFrame({
        'Feature': features,
        'Importância': importancias
    }).sort_values(by='Importância', ascending=False)

    # Plotando a importância das features
    plt.figure(figsize=(10, 6))
    plt.barh(importancia_df['Feature'], importancia_df['Importância'], color='skyblue')
    plt.title('Importância das Features no Modelo de Random Forest')
    plt.xlabel('Importância')
    plt.ylabel('Feature')

    # Adicionando uma anotação sobre as classes
    plt.text(0.05, 0.95, 'Classes: \n0 = normal \n1 = collision \n2 = obstruction', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.show()

# Função para plotar a importância das features usando SHAP
def plotar_importancia_shap(rf, X_train, variaveis):
    # Gerar valores SHAP para a primeira classe (0) ou para todas as classes
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train)

    # Se for um problema multiclasse, shap_values será uma lista, então processamos para a classe de interesse
    if isinstance(shap_values, list):
        # Vamos selecionar a classe 0 (você pode alterar para outra classe, se necessário)
        shap.summary_plot(shap_values[0], X_train, feature_names=variaveis, plot_type="bar")
    else:
        # Caso seja um problema binário, shap_values será um array
        shap.summary_plot(shap_values, X_train, feature_names=variaveis, plot_type="bar")

    # Exibindo o gráfico
    plt.show()

# Carregar os dados
df = carregar_e_processar_dados()

# Calcular a importância das variáveis e resultados de classificação
importancias, classes, clf_report, conf_matrix, rf, X_test, y_test, y_pred, X_train = calcular_importancia_features(df)

# Exibir a importância das variáveis
print("Importância das variáveis no modelo Random Forest:")
for i, feature in enumerate(['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
    print(f'{feature}: {importancias[i]:.4f}')

# Exibir relatório de classificação
print("\nRelatório de Classificação:")
print(clf_report)

# Exibir a matriz de confusão
print("\nMatriz de Confusão:")
print(conf_matrix)

# Plotar a importância das features do Random Forest
plotar_importancia(importancias, classes)

# Plotar a importância das features com SHAP
variaveis = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
plotar_importancia_shap(rf, X_train, variaveis)
