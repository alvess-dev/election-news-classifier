import pandas as pd
import random as rd
import numpy as np


def prepararDfTeste(dfA, dfB):
    dados = []

    # embaralha os dfs
    dfEmbaralhado = pd.concat([dfA, dfB], ignore_index=True)
    indices = list(dfEmbaralhado.index)
    rd.shuffle(indices)
    dfEmbaralhado = dfEmbaralhado.iloc[indices].reset_index(drop=True)

    # cria uma coluna pra tag
    for i in range(len(dfEmbaralhado)):
        news = dfEmbaralhado.iloc[i, 0]
        tag = news[0]
        news = news[1:]
        dados.append([tag, news])

    return pd.DataFrame(dados, columns=['tag', 'noticia'])


def prepararDfsTreino(dfF, dfR):
    for i in range(len(dfF)):
        noticia = dfF.iloc[i, 0]
        noticia.remove('f')

    for i in range(len(dfR)):
        noticia = dfR.iloc[i, 0]
        noticia.remove('r')

    palavras = []
    for i in range(len(dfF)):
        vetor = dfF.iloc[i, 0]
        for word in vetor:
            palavras.append(word)
    dfF = pd.DataFrame({'palavras': palavras})
    dfF = dfF.groupby('palavras').size().reset_index(name='aparicoesFalsas')

    palavras = []
    for i in range(len(dfR)):
        vetor = dfR.iloc[i, 0]
        for word in vetor:
            palavras.append(word)
    dfR = pd.DataFrame({'palavras': palavras})
    dfR = dfR.groupby('palavras').size().reset_index(name='aparicoesReais')

    return dfF, dfR


def evaluate(dfReal, dfFalso, noticiasTeste):
    # Junta os dataframes e une as palavras
    dfJunto = dfReal.merge(dfFalso, how='outer')

    # Conta quantas palavras tem em uma e não tem na outra
    nReal = dfJunto.loc[dfJunto['aparicoesReais'].isna(), 'palavras'].count()
    nFalsa = dfJunto.loc[dfJunto['aparicoesFalsas'].isna(), 'palavras'].count()

    # Preenche as palavras que não aparecem com 1
    dfJunto.fillna(1, inplace=True)

    # Soma as palavras com 0 aparições com 1 para evitar multiplicação por 0
    dfJunto.loc[dfJunto['aparicoesReais'] > 1, 'aparicoesReais'] += 1
    dfJunto.loc[dfJunto['aparicoesFalsas'] > 1, 'aparicoesFalsas'] += 1

    # Soma o total de aparições
    totalReal = dfJunto['aparicoesReais'].sum()
    totalFalso = dfJunto['aparicoesFalsas'].sum()

    # Balanceia a ordem de grandeza das aparições
    dfJunto['aparicoesFalsas'] *= totalReal / totalFalso

    # Faz o coeficiente de bray courtis
    dfJunto['bc'] = abs(dfJunto['aparicoesReais'] - dfJunto['aparicoesFalsas']) / \
        (dfJunto['aparicoesReais'] + dfJunto['aparicoesFalsas'])

    # Limpa palavras com coeficiente menor que 0.3 (palavras com aparições equivalentes em ambos casos)
    dfJunto = dfJunto.loc[dfJunto['bc'] > 0.3]

    # Pega o total de aparições após NLP
    tFalsas = dfJunto['aparicoesFalsas'].sum()
    tReais = dfJunto['aparicoesReais'].sum()

    # Define um df de teste
    dfJuntoTeste = dfJunto

    # Faz os calculos das probabilidades
    dfJuntoTeste['aparicoesFalsas'] = dfJunto['aparicoesFalsas'] / \
        (nFalsa + tFalsas)
    dfJuntoTeste['aparicoesReais'] = dfJunto['aparicoesReais'] / \
        (nReal + tReais)

    # Pega o log na base 10
    dfJuntoTeste['aparicoesFalsas'] = np.log10(dfJuntoTeste['aparicoesFalsas'])
    dfJuntoTeste['aparicoesReais'] = np.log10(dfJuntoTeste['aparicoesReais'])

    # NOTICIA:

    indices = ['fF', 'fR', 'rF', 'rR']
    dfDados = pd.DataFrame(0, index=indices, columns=['num'])

    for i in range(len(noticiasTeste)):
        noticia = noticiasTeste.iloc[i, 1]
        tag = noticiasTeste.iloc[i, 0]

        # Cria um df com a notícia limpa
        dfNoticia = pd.DataFrame(data=noticia, columns=['palavra'])

        # Junta o df da noticia com o das probabilidades
        dfNoticia = dfNoticia.merge(
            dfJuntoTeste, how='inner', left_on='palavra', right_on='palavras')

        # Faz a soma das probabilidades
        iFalsa = dfNoticia['aparicoesFalsas'].sum()
        iReal = dfNoticia['aparicoesReais'].sum()

        # Confere o resultado
        if (iFalsa > iReal):  # Falsa
            resultado = 'F'
        else:
            resultado = 'R'

        chave = tag + resultado
        dfDados.loc[chave, 'num'] += 1

    acuracia = round(float(
        (dfDados.loc['fF', 'num'] + dfDados.loc['rR', 'num']) / dfDados['num'].sum()), 4)

    sensiReal = round(float(
        dfDados.loc['rR', 'num'] / (dfDados.loc['rR', 'num'] + dfDados.loc['rF', 'num'])), 4)
    sensiFalsa = round(float(
        dfDados.loc['fF', 'num'] / (dfDados.loc['fF', 'num'] + dfDados.loc['fR', 'num'])), 4)

    precReal = round(float(
        dfDados.loc['rR', 'num'] / (dfDados.loc['rR', 'num'] + dfDados.loc['fR', 'num'])), 4)
    precFalsa = round(float(
        dfDados.loc['fF', 'num'] / (dfDados.loc['fF', 'num'] + dfDados.loc['rF', 'num'])), 4)

    print(dfDados)
    return acuracia, sensiReal, precReal, sensiFalsa, precFalsa


dfFalsoTreino = pd.read_json(
    'data/train/fakeTrain.json', orient='records', lines=True)
dfRealTreino = pd.read_json(
    'data/train/realTrain.json', orient='records', lines=True)

dfFalsoTeste = pd.read_json(
    'data/test/fakeTest.json', orient='records', lines=True)
dfRealTeste = pd.read_json(
    'data/test/realTest.json', orient='records', lines=True)

dfTestes = prepararDfTeste(dfFalsoTeste, dfRealTeste)

dfFalsoTreino, dfRealTreino = prepararDfsTreino(dfFalsoTreino, dfRealTreino)

acuracia, sensiReal, precReal, sensiFalsa, precFalsa = evaluate(
    dfRealTreino, dfFalsoTreino, dfTestes)

# Dicionário com os dados
dados_estatisticas = {
    'Métrica': ['Acurácia', 'Sensibilidade Reais', 'Precisão Reais', 'Sensibilidade Falsas', 'Precisão Falsas'],
    'Valor': list([
        acuracia,
        sensiReal,
        precReal,
        sensiFalsa,
        precFalsa
    ])
}

# Cria o DataFrame
dfEstatisticas = pd.DataFrame(dados_estatisticas)

print(dfEstatisticas)
