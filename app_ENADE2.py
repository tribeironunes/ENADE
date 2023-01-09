import streamlit as st
import pandas as pd
from pandas import factorize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from math import sqrt
import plotly.graph_objects as go
import plotly.express as px


def get_data(filename):
    enade = pd.read_csv(filename)
    return enade

header = st.container()
dataset = st.container()
features = st.container()
predictions = st.container()

with header:
    st.title('Universidade Federal de Catalão')
    st.subheader('Exame Nacional de Desempenho dos Estudantes')

with dataset:
    # lendo o dataset
    ENADE = get_data('ENADE.csv')

    # adiciona a coluna "taxa_participacao" ao DataFrame
    ENADE['taxa_participacao'] = ENADE['presentes'] / ENADE['populacao']

with features:
    st.sidebar.write('**PROGRAD/UFCAT**')
    st.sidebar.caption('**Esta aplicação permite explorar os dados do ENADE e '
             'visualizar gráficos sobre a performance dos cursos da nossa instituição. '
             'Selecione uma das opções no menu lateral para começar a explorar os dados.**')

    st.sidebar.write("**Exibir o conjunto de dados ou a estatística descritiva do conjunto de dados:**")
    # exibe o dataset
    if st.sidebar.button("Conjunto de dados"):
        # exibe os dados gerais
        st.subheader('Visão geral dos dados:')
        st.dataframe(ENADE, width=1000)
        st.caption('Fonte: https://enade.inep.gov.br/enade/#!/relatorioCursos')
        st.caption('*Até 2014 os relatórios do ENADE não discriminam os dados por cidade.')
        st.caption('**Os dados de 2015 estão indiponíveis para a cidade de Catalão.')

    # exibe a estatística descritiva do dataset
    if st.sidebar.button("Estatística descritiva"):
        # descreve os dados
        st.subheader("Estatística descritiva do conjunto de dados do ENADE - UFCAT:")
        st.table(ENADE.describe())

    # seleciona uma das colunas do dataset para gerar o histograma
    st.sidebar.write("**Gerar Histograma e Box Plot da variável selecionada:**")
    selected_column = st.sidebar.selectbox('Selecione uma variável', ENADE.columns)

    st.subheader("Gráficos gerados pela aplicação:")

    # Usa o seaborn para criar o histograma
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(ENADE[selected_column], ax=ax)
    ax.set_xlabel(selected_column, fontsize=12)
    ax.set_ylabel("Frequência", fontsize=12)

    # exibe o histograma
    st.write("**Histograma da variável selecionada**")
    st.pyplot(fig)
    st.caption("*Um histograma é um gráfico que mostra a distribuição de um conjunto de dados. "
               "Ele é útil para visualizar padrões e tendências nos dados e "
               "pode ser usado para comparar as distribuições de diferentes grupos ou variáveis.")

    # cria o box plot
    st.write("**Box plot da variável selecionada**")
    fig = px.box(ENADE, y=selected_column)
    st.plotly_chart(fig)
    st.caption("*Um box plot é uma representação gráfica de dados que mostra a distribuição de um conjunto de dados"
               " através de quartis. Ele é usado para identificar outliers (valores atípicos)"
               " e para comparar a distribuição de dados entre grupos. "
               "O box plot é composto por uma caixa que representa os primeiros três quartis, "
               "uma linha no meio da caixa que representa a mediana e dois bigodes que representam "
               "os valores máximos e mínimos, exceto os outliers.")

    st.write("**Gráfico de dispersão das variáveis selecionadas**")
    # seleciona duas colunas do dataset para gerar o gráfigo de dispersão
    st.sidebar.write("**Gerar Gráfico de Dispersão das variáveis selecionadas:**")
    selected_column_x = st.sidebar.selectbox('Selecione uma variável para o eixo x', ENADE.columns)
    selected_column_y = st.sidebar.selectbox('Selecione uma variável para o eixo y', ENADE.columns)
    # cria a figura
    fig = px.scatter(ENADE, x=selected_column_x, y=selected_column_y)
    # exibe a figura
    st.plotly_chart(fig)
    st.caption("*O gráfico de dispersão é usado para visualizar a relação entre duas variáveis "
             "quantitativas e identificar padrões e tendências. Ele pode ser usado para comparar "
             "relações entre várias variáveis e para identificar correlação linear ou não linear e outliers.")

    st.sidebar.write("**Gerar gráficos adicionais:**")
    if st.sidebar.button("Pair Plot"):
        # cria o pair plot
        st.write("**Pair plot**")
        sns.pairplot(ENADE, hue='ano')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.caption(
            "*O pair plot é um gráfico de dispersão que mostra a relação entre todas as variáveis de um conjunto de dados. "
            "É útil para visualizar a correlação entre as variáveis e para identificar padrões e tendências nos dados. "
            "Também pode ser usado para comparar as relações entre várias variáveis.")

    if st.sidebar.button("Mapa de Calor"):
        # cria o mapa de calor da matriz de correlação das colunas do banco de dados
        st.write("**Mapa de calor da matriz de correlação das colunas do banco de dados:**")
        corr = ENADE.corr()
        sns.heatmap(corr, annot=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.caption("*O Mapa de Calor da Matriz de Correlação é um tipo de gráfico que permite visualizar "
                   "a correlação entre as colunas de um banco de dados. "
                   "Cada célula do mapa representa a correlação entre duas colunas, "
                   "sendo que a cor da célula varia de acordo com o valor da correlação. "
                   "Isso permite identificar facilmente quais colunas estão fortemente correlacionadas"
                   " e quais apresentam correlação mais fraca ou nula.")

    st.subheader('Examinando o Conceito e a Taxa de Participação por Curso')
    st.sidebar.write('**Conceito & Taxa de Participação por Curso:**')
    # Adicionando uma nova opção de seleção para o usuário escolher um curso
    selected_course = st.sidebar.selectbox('Selecione o curso:', ENADE['cursos'].unique())
    # Exibindo o ano selecionado pelo usuário
    st.write('O curso selecionado foi:', selected_course)
    # Filtrando o conjunto de dados pelo curso selecionado
    conceito_curso_historico = ENADE[ENADE['cursos'] == selected_course]
    # Criando uma nova coluna chamada "taxa_participacao" no conjunto de dados "conceito_curso_historico"
    conceito_curso_historico['taxa_participacao'] = conceito_curso_historico['presentes'] / conceito_curso_historico[
        'populacao']

    # Define o nível de confiança desejado (95%)
    confianca = 0.95
    # Calcula o tamanho da amostra
    n = conceito_curso_historico.shape[0]
    # Calcula o valor de z para o nível de confiança escolhido
    z = 1.96
    # Calcula a média do conceito de curso
    media_conceito = conceito_curso_historico['conceito'].mean()
    # Calcula o desvio padrão do conceito de curso
    desvio_conceito = conceito_curso_historico['conceito'].std()
    # Calcula o intervalo de confiança para a média do conceito de curso
    ic_conceito = media_conceito + z * desvio_conceito / sqrt(n)
    # Calcula a média da taxa de participação
    media_taxa = conceito_curso_historico['taxa_participacao'].mean()
    # Calcula o desvio padrão da taxa de participação
    desvio_taxa = conceito_curso_historico['taxa_participacao'].std()
    # Calcula o intervalo de confiança para a média da taxa de participação
    ic_taxa = media_taxa + z * desvio_taxa / sqrt(n)

    # Exibindo informações adicionais sobre os dados
    st.write("- Número de participações no ENADE: ", '{:.2f}'.format(conceito_curso_historico.shape[0]))
    st.write("- Média do conceito de curso: ", '{:.2f}'.format(conceito_curso_historico['conceito'].mean()))
    st.write("- Desvio padrão do conceito de curso: ", '{:.2f}'.format(conceito_curso_historico['conceito'].std()))
    st.write("- Intervalo de confiança da média do conceito de curso: ", '{:.2f}'.format(ic_conceito))
    st.write("- Taxa média de participação: ", '{:.2f}'.format(conceito_curso_historico['taxa_participacao'].mean()))
    st.write("- Desvio padrão da taxa de participação: ", '{:.2f}'.format(conceito_curso_historico['taxa_participacao'].std()))
    st.write("- Intervalo de confiança da média da taxa de participação: ", '{:.2f}'.format(ic_taxa))

    # Calculando o coeficiente de correlação de Spearman
    corr, _ = spearmanr(conceito_curso_historico['taxa_participacao'], conceito_curso_historico['conceito'])
    st.write('- Correlação entre o conceito de curso e a taxa de participação: ', '{:.2f}'.format(corr))

    # Adicionando a opção de exibir o tutorial
    exibir_tutorial = st.checkbox("Clique para exibir as definições")
    if exibir_tutorial:
        st.caption(
            "1. Média: valor numérico que indica a tendência central de um conjunto de dados, "
            "calculado como a soma de todos os valores dividida pelo número de valores;\n"
            "2. Desvio padrão da média: medida da variabilidade dos resultados de uma amostra em relação à média da população, "
            "calculado como o desvio padrão da amostra dividido pelo radical quadrado do tamanho da amostra;\n"
            "3. Intervalo de confiança da média: estimativa da precisão da média de uma amostra em relação à média da população, "
            "calculado com base no tamanho da amostra, o desvio padrão da amostra e o nível de confiança escolhido;\n"
            "4. O coeficiente de correlação de Spearman é uma medida de associação entre duas variáveis ordinais."
            " Ele varia de -1 a 1, sendo que valores próximos de -1 indicam uma correlação negativa forte, valores "
            "próximos de 0 indicam uma correlação fraca ou ausência de correlação, "
            "e valores próximos de 1 indicam uma correlação positiva forte.\n")

    st.subheader('Examinando o Conceito e a Taxa de Participação por Edição do ENADE')
    st.sidebar.write('**Conceito & Taxa de Participação por Edição:**')
    # Adicionando o campo de seleção
    selected_year_1 = st.sidebar.selectbox('Escolha a edição:', ENADE['ano'].unique())
    # Exibindo o ano selecionado pelo usuário
    st.write('A edição selecionada foi:', selected_year_1)

    # Filtrando o conjunto de dados pelo ano selecionado pelo usuário
    conceito_curso_historico = ENADE[ENADE['ano'] == selected_year_1]
    st.write('Conceito de curso:')
    st.bar_chart(conceito_curso_historico.pivot_table(index='cursos', columns='ano', values='conceito'))

    comparecimento_prova = conceito_curso_historico.pivot_table(index='cursos', columns='ano',
                                                                values='taxa_participacao')

    # Remove linhas com valores NaN ou inf do conjunto de dados
    ENADE = ENADE.dropna(how="any")
    ENADE = ENADE[~ENADE.isin([np.nan, np.inf, -np.inf]).any(1)]

    # passa os tipos string para numéricos
    ENADE['cursos'], _ = factorize(ENADE['cursos'])
    ENADE['grau_academico'], _ = factorize(ENADE['grau_academico'])

    # Preenche valores faltantes com a média das colunas correspondentes
    ENADE = ENADE.fillna(ENADE.mean())

    # Normaliza todas as colunas do conjunto de dados
    for col in ENADE.columns:
        ENADE[col] = (ENADE[col] - ENADE[col].min()) / (ENADE[col].max() - ENADE[col].min())

    # Gera o gráfico de linhas
    st.write('Taxa de participação:')
    st.line_chart(comparecimento_prova)

    st.caption("Diretoria de Currículo, Avaliação e Diploma (PROGRAD/UFCAT)")

    # Adiciona uma mensagem de aviso na página inicial
    st.warning('Esta é uma versão Beta da aplicação e pode apresentar erros ou inconsistências')
    # Adiciona a badge "Beta" na barra lateral
    st.sidebar.warning('Esta é uma versão Beta da aplicação')
