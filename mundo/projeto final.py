import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, roc_curve, auc, silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram  # Adicionados para dendrograma e linkage
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    return pd.read_csv('transferencias_jogadores.csv')

def load_data2():
    uploaded_file = st.file_uploader("Carregar arquivo .CSV (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Por favor, carregue um arquivo CSV para começar.")
        return None


# Função para exibir o gráfico dos 5 jogadores com mais tempo jogado
def exibir_top_5_jogadores_tempo():
    dados_transferencias = pd.read_csv('transferencias_jogadores.csv')
    if not pd.api.types.is_numeric_dtype(dados_transferencias['minutes played']):
        dados_transferencias['minutes played'] = pd.to_numeric(dados_transferencias['minutes played'], errors='coerce')
    dados_transferencias = dados_transferencias.dropna(subset=['minutes played'])
    top_jogadores_tempo = dados_transferencias.nlargest(5, 'minutes played')
    st.write("Os 5 jogadores com mais tempo jogado:")
    st.write(top_jogadores_tempo[['name', 'team', 'minutes played']])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_jogadores_tempo, x='minutes played', y='name', palette='coolwarm')
    plt.title("Top 5 Jogadores com Mais Tempo Jogado", fontsize=16)
    plt.xlabel("Minutos Jogados", fontsize=12)
    plt.ylabel("Jogador", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Função para exibir o valor total das transferências por país
def exibir_valor_por_pais():
    dados_transferencias = pd.read_csv('transferencias_jogadores.csv')
    dados_cruzamento = pd.read_excel('cruzamento.xlsx')
    dados_transferencias['current_value'] = dados_transferencias['current_value'].replace(
        {'M': '*1e6', 'K': '*1e3'}, regex=True).map(pd.eval).astype(float)
    dados_agrupados = pd.merge(dados_transferencias, dados_cruzamento, on='team', how='inner')
    valor_por_pais = dados_agrupados.groupby('country')['current_value'].sum()
    valor_por_pais = valor_por_pais.sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    valor_por_pais.head(10).plot(kind='bar', color='skyblue')
    plt.title('Top 10 Países com Maior Valor de Mercado de Transferências', fontsize=16)
    plt.xlabel('País', fontsize=12)
    plt.ylabel('Valor Total das Transferências (em milhões)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.write(valor_por_pais.head(10))
    st.pyplot(plt)

# Função para modelagem e previsão de novos dados
def modelo_machine_learning():
    st.write("### Modelo de Machine Learning - Análise de Transferências")
    df = load_data()

    # Seleção de 20 linhas aleatórias para mostrar ao usuário
    df_sampled = df.sample(n=20, random_state=42)
    st.write("Amostra de Dados (20 Linhas Selecionadas Aleatoriamente):")
    st.write(df_sampled)

    # Seleção de alvo e preditores fora da sidebar, com armazenamento em session_state
    if 'target' not in st.session_state:
        st.session_state.target = df_sampled.columns[0]
    if 'features' not in st.session_state:
        st.session_state.features = []

    st.write("### Escolha as variáveis para o modelo de Machine Learning:")

    # Seleção de variável alvo e preditoras a partir de session_state para manter as opções
    target = st.selectbox("Escolha a variável alvo:", df_sampled.columns, index=df_sampled.columns.get_loc(st.session_state.target))
    features = st.multiselect("Escolha as variáveis preditoras:", df_sampled.columns.drop(target), default=st.session_state.features)

    # Atualizando session_state com as seleções
    st.session_state.target = target
    st.session_state.features = features

    if not features or not target:
        st.warning("Selecione uma variável alvo e pelo menos uma preditora para continuar.")
    else:
        # Pré-processamento
        X = df_sampled[features]
        y = df_sampled[target]
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modelos
        models = {
            "Naive Bayes": GaussianNB(),
            "k-NN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(probability=True),
            "Logistic Regression": LogisticRegression(),
            "MLP": MLPClassifier(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Stacking": StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('lr', LogisticRegression())], final_estimator=LogisticRegression())
        }

        # Seleção de modelo fora da sidebar
        selected_model = st.selectbox("Escolha o modelo", list(models.keys()))
        model = models[selected_model]

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        st.write(f"### Modelo Selecionado: {selected_model}")
        st.write(f"- **Accuracy**: {accuracy:.2f}")
        st.write(f"- **F1-Score**: {f1:.2f}")
        st.write(f"- **Precision**: {precision:.2f}")

        # Matriz de Confusão
        st.write("### Matriz de Confusão")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        # Curvas ROC
        if y_proba is not None:
            st.write("### Curva ROC e AUC")
            fig, ax = plt.subplots()
            for i in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
                ax.plot(fpr, tpr, label=f"Classe {i} (AUC = {auc(fpr, tpr):.2f})")
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("Curva ROC")
            ax.legend()
            st.pyplot(fig)

        # Previsão de Novos Dados
        st.write("### Previsão de Novos Dados")
        new_data = [st.number_input(f"Insira o valor de {col}", value=0.0) for col in features]
        if st.button("Classificar Novos Dados"):
            new_data_scaled = scaler.transform([new_data])
            new_pred = model.predict(new_data_scaled)[0]
            new_proba = model.predict_proba(new_data_scaled)[0] if y_proba is not None else None
            st.write("### Resultado da Classificação")
            st.write(f"**Classe Predita**: {new_pred}")
            if new_proba is not None:
                st.write("### Probabilidades por Classe")
                for i, prob in enumerate(new_proba):
                    st.write(f"- Classe {i}: {prob:.2f}")

def exibir_cluster():
    # Carregar os dados
    df = load_data()

    # Selecionar 20 linhas aleatórias
    df_sampled = df.sample(n=20, random_state=42)

    # Exibição dos dados
    st.title("Clustering Dashboard")
    st.write("### Amostra de Dados (20 Linhas Selecionadas Aleatoriamente):")
    st.write(df_sampled)

    # Configuração da Sidebar
    st.sidebar.header("Configurações")
    if st.sidebar.checkbox("Mostrar Conjunto de Dados Original"):
        st.write("### Dados Originais")
        st.write(df)

    # Seleção de colunas para clustering
    st.sidebar.subheader("Seleção de Variáveis")
    features = st.sidebar.multiselect("Escolha as Variáveis para Clustering", df_sampled.columns)

    if not features:
        st.warning("Selecione pelo menos duas variáveis para clustering.")
    else:
        # Filtrar colunas numéricas
        data = df_sampled[features].select_dtypes(include=[np.number])

        if data.shape[1] < len(features):
            st.warning("Algumas das colunas selecionadas não são numéricas e foram ignoradas.")

        # Verificar se há pelo menos duas colunas numéricas
        if data.shape[1] < 2:
            st.error("É necessário selecionar pelo menos duas colunas numéricas.")
        else:
            # Normalização dos dados
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Método do Cotovelo
            st.sidebar.subheader("Método do Cotovelo")
            elbow_show = st.sidebar.checkbox("Calcular e Mostrar Cotovelo")
            if elbow_show:
                st.write("### Método do Cotovelo")
                inertia = []
                range_clusters = range(1, 11)
                for k in range_clusters:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(data_scaled)
                    inertia.append(kmeans.inertia_)
                
                fig, ax = plt.subplots()
                ax.plot(range_clusters, inertia, marker='o', linestyle='--')
                ax.set_xlabel("Número de Clusters")
                ax.set_ylabel("Inércia")
                ax.set_title("Método do Cotovelo")
                st.pyplot(fig)
                
            # Escolha do número de clusters
            n_clusters = st.sidebar.slider("Escolha o Número de Clusters (K-Means e Hierarchical)", 2, 10, 3)

            # Escolha do algoritmo
            st.sidebar.subheader("Configuração de Clustering")
            algorithm = st.sidebar.selectbox("Escolha o Algoritmo", ["K-Means", "Hierarchical Clustering", "DBSCAN"])

            # Parâmetros do DBSCAN
            if algorithm == "DBSCAN":
                eps = st.sidebar.slider("Escolha o Valor de eps", 0.1, 1.5, 0.5)
                min_samples = st.sidebar.slider("Escolha o Mínimo de Amostras", 1, 10, 5)

            # Aplicação do Clustering
            st.write(f"### Resultado do Clustering com {algorithm}")
            if algorithm == "K-Means":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(data_scaled)
                df_sampled['Cluster'] = clusters
                centroids = kmeans.cluster_centers_
                st.write(f"Silhouette Score: {silhouette_score(data_scaled, clusters):.2f}")
                
                # Gráfico de clusters
                fig, ax = plt.subplots()
                sns.scatterplot(data=df_sampled, x=features[0], y=features[1], hue='Cluster', palette="Set2", ax=ax)
                ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c="red", label="Centroids")
                ax.legend()
                ax.set_title("K-Means Clustering")
                st.pyplot(fig)

            elif algorithm == "Hierarchical Clustering":
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = hierarchical.fit_predict(data_scaled)
                df_sampled['Cluster'] = clusters
                st.write(f"Silhouette Score: {silhouette_score(data_scaled, clusters):.2f}")
                
                # Dendrograma
                linkage_matrix = linkage(data_scaled, method="ward")
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(linkage_matrix, truncate_mode="lastp", p=10, ax=ax)
                ax.set_title("Dendrograma (Hierarchical Clustering)")
                st.pyplot(fig)
                
                # Gráfico de clusters
                fig, ax = plt.subplots()
                sns.scatterplot(data=df_sampled, x=features[0], y=features[1], hue='Cluster', palette="Set2", ax=ax)
                ax.set_title("Hierarchical Clustering")
                st.pyplot(fig)

            elif algorithm == "DBSCAN":
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(data_scaled)
                df_sampled['Cluster'] = clusters
                unique_clusters = np.unique(clusters)
                n_clusters = len(unique_clusters[unique_clusters != -1])
                st.write(f"Número de Clusters Encontrados (sem ruído): {n_clusters}")
                st.write(f"Silhouette Score: {silhouette_score(data_scaled, clusters):.2f}" if n_clusters > 1 else "Silhouette Score não aplicável.")
                
                # Gráfico de clusters
                fig, ax = plt.subplots()
                sns.scatterplot(data=df_sampled, x=features[0], y=features[1], hue='Cluster', palette="Set2", ax=ax)
                ax.set_title("DBSCAN Clustering")
                st.pyplot(fig)
        
def exibir_pca():
    # Carregar os dados
    df = load_data2()
    st.title("Dashboard de Redução de Dimensionalidade com PCA - Transferências de Jogadores")
    st.markdown("Este dashboard permite aplicar PCA em um conjunto de dados de transferências de jogadores e visualizar os componentes principais.")

    # Exibir o conjunto de dados completo
    st.sidebar.header("Configurações")
    st.sidebar.write("### Pré-visualização dos dados")
    if st.sidebar.checkbox("Mostrar Conjunto de Dados"):
        st.write("### Conjunto de Dados")
        st.write(df)  # Mostra o DataFrame inteiro

    # Utilizar o conjunto de dados completo
    st.write("### Conjunto de Dados Completo")
    st.write(df)  # Exibe o conjunto de dados completo na página

    # Seleção do número de componentes principais
    st.sidebar.write("### Configuração do PCA")
    
    # Filtrando apenas as colunas numéricas para o PCA
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Seleciona colunas numéricas
    if len(num_cols) == 0:
        st.error("O conjunto de dados não contém colunas numéricas para aplicar PCA.")
    else:
        n_components = st.sidebar.slider("Número de Componentes Principais", 1, len(num_cols), 2)

        # Normalização dos dados
        st.sidebar.write("### Normalização dos Dados")
        normalize = st.sidebar.checkbox("Normalizar os Dados (StandardScaler)", value=True)
        if normalize:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[num_cols])
        else:
            df_scaled = df[num_cols].values

        # Aplicar PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)

        # Criar DataFrame com os componentes principais
        pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

        # Tentar identificar a coluna "target" (se houver)
        target_column = None
        for col in df.columns:
            if df[col].dtype == 'object':  # Verifica se a coluna é categórica
                target_column = col
                break

        if target_column:
            pca_df['target'] = df[target_column]
        else:
            pca_df['target'] = "Não Definido"  # Caso não haja coluna target, apenas para visualização

        # Visualizar a variância explicada
        st.sidebar.write("### Variância Explicada")
        if st.sidebar.checkbox("Mostrar Variância Explicada"):
            explained_variance = pca.explained_variance_ratio_ * 100
            st.write("### Variância Explicada por Componente")
            for i, var in enumerate(explained_variance):
                st.write(f"PC{i+1}: {var:.2f}%")
            fig, ax = plt.subplots()
            ax.bar([f"PC{i+1}" for i in range(len(explained_variance))], explained_variance)
            ax.set_title("Variância Explicada")
            ax.set_ylabel("Porcentagem (%)")
            ax.set_xlabel("Componentes Principais")
            st.pyplot(fig)

        # Gráficos de PCA
        st.write("### Visualização dos Componentes Principais")
        chart_type = st.selectbox("Escolha o tipo de gráfico", ["2D Scatterplot", "Heatmap de Correlação"])

        if chart_type == "2D Scatterplot" and n_components >= 2:
            st.write("#### Scatterplot dos Dois Primeiros Componentes Principais")
            plt.figure(figsize=(10, 6))
            scatter = sns.scatterplot(
                x="PC1", y="PC2", hue="target", style="target", 
                data=pca_df, palette="Set1", markers=["o", "s", "D"], alpha=0.7
            )
            plt.title("PCA - Dois Primeiros Componentes")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            
            # Ajustar a posição da legenda para fora do gráfico
            plt.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            st.pyplot(plt.gcf())

        elif chart_type == "Heatmap de Correlação":
            st.write("#### Heatmap da Correlação dos Componentes Principais")
            # Apenas calcular a correlação entre as colunas numéricas
            corr_matrix = pca_df.select_dtypes(include=['float64', 'int64']).corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt.gcf())

        # Download dos Componentes Principais
        st.write("### Download dos Resultados do PCA")
        csv = pca_df.to_csv(index=False)
        st.download_button(
            label="Realizar Download dos Componentes Principais como CSV",
            data=csv,
            file_name="pca_components.csv",
            mime="text/csv",
        )

def exibir_ra():
    # Carregar os dados
    df = load_data2()
    st.write("### Conjunto de Dados - Transferências de Jogadores")
    st.dataframe(df)

    # Selecionar aleatoriamente 20 linhas do dataframe (se houver 20 ou mais linhas)
    if len(df) >= 20:
        df_sampled = df.sample(n=20, random_state=42)  # 20 linhas aleatórias
    else:
        df_sampled = df  # Se houver menos de 20 linhas, usamos o DataFrame completo

    st.write("### Amostra Aleatória de 20 Linhas (ou menos se o conjunto de dados for pequeno)")
    st.dataframe(df_sampled)

    # Converter para formato binário
    st.write("### Dados de Transações Binárias")
    try:
        # Ajuste: Cada valor não nulo/zero será tratado como 1
        binary_df = df_sampled.applymap(lambda x: 1 if x != 0 else 0)
        st.dataframe(binary_df)
    except Exception as e:
        st.error(f"Erro ao transformar os dados em binário: {e}")
        st.stop()

    # Aplicar Apriori
    st.write("### Regras de Associação com Apriori")
    try:
        frequent_items_apriori = apriori(binary_df, min_support=0.5, use_colnames=True)
        st.write("Itens Frequentes (Apriori):")
        st.dataframe(frequent_items_apriori)

        # Gerar regras de associação usando Apriori
        rules_apriori = association_rules(frequent_items_apriori, metric="confidence", min_threshold=0.7, num_itemsets=len(frequent_items_apriori))
        st.write("Regras de Associação (Apriori):")
        st.dataframe(rules_apriori)
    except ValueError as e:
        st.error(f"Erro ao aplicar Apriori: {e}")

    # Aplicar FP-Growth
    st.write("### Regras de Associação com FP-Growth")
    try:
        frequent_items_fpgrowth = fpgrowth(binary_df, min_support=0.5, use_colnames=True)
        st.write("Itens Frequentes (FP-Growth):")
        st.dataframe(frequent_items_fpgrowth)

        # Gerar regras de associação usando FP-Growth
        rules_fpgrowth = association_rules(frequent_items_fpgrowth, metric="confidence", min_threshold=0.7, num_itemsets=len(frequent_items_fpgrowth))
        st.write("Regras de Associação (FP-Growth):")
        st.dataframe(rules_fpgrowth)
    except ValueError as e:
        st.error(f"Erro ao aplicar FP-Growth: {e}")

def exibir_reg():
    # Carregar os dados
    df = load_data2()
    st.write("### Conjunto de Dados - Transferências de Jogadores")
    st.dataframe(df)

    # Escolher uma variável dependente 'y' (por exemplo, 'valor_transferencia') e uma independente 'X' (por exemplo, 'idade', 'salario', etc.)
    # Ajuste conforme a estrutura do seu CSV
    # Certifique-se de que as colunas de interesse estão no seu arquivo CSV
    if 'current_value' in df.columns and 'age' in df.columns:
        X = df[['age']].values  # Substitua por outras variáveis se necessário
        y = df['current_value'].values
    else:
        st.error("As colunas 'valor_transferencia' ou 'idade' não foram encontradas no arquivo.")
        st.stop()

    # Dividir o conjunto de dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Regressão Linear
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)

    # Regressão Polinomial (Grau 2)
    poly_features = PolynomialFeatures(degree=2)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    y_pred_poly = poly_model.predict(X_poly_test)

    # Métricas de avaliação
    def calculate_metrics(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, mae, rmse

    metrics_linear = calculate_metrics(y_test, y_pred_linear)
    metrics_poly = calculate_metrics(y_test, y_pred_poly)

    # Dashboard
    st.write("### Gráficos de Regressão")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de regressão linear
    ax[0].scatter(X_test, y_test, label="Dados Reais", color="blue")
    ax[0].plot(X_test, y_pred_linear, label="Regressão Linear", color="red")
    ax[0].set_title("Regressão Linear")
    ax[0].set_xlabel("age")
    ax[0].set_ylabel("current_value")
    ax[0].legend()

    # Gráfico de regressão polinomial
    ax[1].scatter(X_test, y_test, label="Dados Reais", color="blue")
    ax[1].scatter(X_test, y_pred_poly, label="Regressão Polinomial", color="green", alpha=0.7)
    ax[1].set_title("Regressão Polinomial (Grau 2)")
    ax[1].set_xlabel("age")
    ax[1].set_ylabel("current_value")
    ax[1].legend()

    st.pyplot(fig)

    # Métricas de avaliação
    st.write("### Métricas de Avaliação")
    metrics_df = pd.DataFrame({
        "Modelo": ["Regressão Linear", "Regressão Polinomial (Grau 2)"],
        "R²": [metrics_linear[0], metrics_poly[0]],
        "MAE": [metrics_linear[1], metrics_poly[1]],
        "RMSE": [metrics_linear[2], metrics_poly[2]]
    })
    st.dataframe(metrics_df)

    # Previsão para novos dados
    st.sidebar.header("Previsão com Novos Dados")
    new_data = st.sidebar.number_input("Insira o valor de Idade para prever o Valor da Transferência", value=0.0)
    if st.sidebar.button("Prever"):
        # Previsão com regressão linear
        linear_prediction = linear_model.predict([[new_data]])[0]

        # Previsão com regressão polinomial
        poly_prediction = poly_model.predict(poly_features.transform([[new_data]]))[0]

        st.sidebar.write("### Resultados da Previsão")
        st.sidebar.write(f"**Regressão Linear**: Valor da Transferência = {linear_prediction:.2f}")
        st.sidebar.write(f"**Regressão Polinomial (Grau 2)**: Valor da Transferência = {poly_prediction:.2f}")


# Título do dashboard
st.title('Dashboard de Análise de Transferências de Jogadores')

# Sidebar com opções
st.sidebar.title("Escolha a Análise")

# Definindo a opção selecionada para garantir que não seja desmarcada até uma nova seleção
if 'sidebar_option' not in st.session_state:
    st.session_state.sidebar_option = 'Modelo de Machine Learning'

# Botões para selecionar a opção e manter o estado
if st.sidebar.button('Top 5 Jogadores com Mais Tempo Jogado'):
    st.session_state.sidebar_option = 'Top 5 Jogadores com Mais Tempo Jogado'

if st.sidebar.button('Valor Total das Transferências por País'):
    st.session_state.sidebar_option = 'Valor Total das Transferências por País'

if st.sidebar.button('Modelo de Machine Learning'):
    st.session_state.sidebar_option = 'Modelo de Machine Learning'

if st.sidebar.button('Modelo de cluster'):
    st.session_state.sidebar_option = 'Modelo de cluster'

if st.sidebar.button('Modelo de pca'):
    st.session_state.sidebar_option = 'Modelo de PCA'

if st.sidebar.button('Modelo de RA'):
    st.session_state.sidebar_option = 'Modelo de RA'

if st.sidebar.button('Modelo de REG'):
    st.session_state.sidebar_option = 'Modelo de REG'

# Executando a função baseada na opção selecionada
if st.session_state.sidebar_option == 'Top 5 Jogadores com Mais Tempo Jogado':
    st.subheader("Top 5 Jogadores com Mais Tempo Jogado")
    exibir_top_5_jogadores_tempo()

elif st.session_state.sidebar_option == 'Valor Total das Transferências por País':
    st.subheader("Valor Total das Transferências por País")
    exibir_valor_por_pais()

elif st.session_state.sidebar_option == 'Modelo de Machine Learning':
    st.subheader("Modelo de Machine Learning")
    modelo_machine_learning()

elif st.session_state.sidebar_option == 'Modelo de cluster':
    st.subheader("Modelo de cluster")
    exibir_cluster()

elif st.session_state.sidebar_option == 'Modelo de PCA':
    st.subheader("Modelo de PCA")
    exibir_pca()
    
elif st.session_state.sidebar_option == 'Modelo de RA':
    st.subheader("Modelo de RA")
    exibir_ra()

elif st.session_state.sidebar_option == 'Modelo de REG':
    st.subheader("Modelo de REG")
    exibir_reg()