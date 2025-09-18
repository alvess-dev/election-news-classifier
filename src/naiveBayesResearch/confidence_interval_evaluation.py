import pandas as pd
import random as rd
import numpy as np

accuracies = []
sensitivitiesReal = []
sensitivitiesFake = []
precisionsReal = []
precisionsFake = []

dfFakeOriginal = pd.read_json(
    '../data/train/fakeTrain.json', orient='records', lines=True)
dfRealOriginal = pd.read_json(
    '../data/train/realTrain.json', orient='records', lines=True)


def splitTrainTest(df):
    dfTest = pd.DataFrame()
    dfTrain = pd.DataFrame()
    for i in range(len(df)):
        newsVector = df.iloc[i, 0]
        if rd.random() > 0.8:
            dfTest = pd.concat([dfTest, pd.DataFrame(
                [[newsVector]], columns=['news'])], ignore_index=True)
        else:
            dfTrain = pd.concat([dfTrain, pd.DataFrame(
                [[newsVector]], columns=['news'])], ignore_index=True)
    return dfTest, dfTrain


def prepareTestDf(dfFake, dfReal):
    dfShuffled = pd.concat([dfFake, dfReal], ignore_index=True)
    indexes = list(dfShuffled.index)
    rd.shuffle(indexes)

    tags = []
    newsContent = []

    for i in indexes:
        news = dfShuffled.iloc[i, 0]
        tags.append(news[0])
        newsContent.append(news[1:])

    return pd.DataFrame({'tag': tags, 'news': newsContent})


def prepareTrainDfs(dfFake, dfReal):
    fakeWords = []
    for i in range(len(dfFake)):
        news = dfFake.iloc[i, 0][1:]
        fakeWords.extend(news)
    dfFake = pd.DataFrame({'word': fakeWords})
    dfFake = dfFake.groupby('word').size().reset_index(name='fakeCount')

    realWords = []
    for i in range(len(dfReal)):
        news = dfReal.iloc[i, 0][1:]
        realWords.extend(news)
    dfReal = pd.DataFrame({'word': realWords})
    dfReal = dfReal.groupby('word').size().reset_index(name='realCount')

    return dfFake, dfReal

# def thresholdsCut(df):
# aqui vai a logica de filtrar por threshold, pra cada coeficiente

def probabilitiesCalculation(df, countFakeUnique, countRealUnique):
    totalFilteredFake = df['fakeCount'].sum()
    totalFilteredReal = df['realCount'].sum()

    dfProbabilities = df.copy()
    dfProbabilities['fakeCount'] = df['fakeCount'] / \
        (countFakeUnique + totalFilteredFake)
    dfProbabilities['realCount'] = df['realCount'] / \
        (countRealUnique + totalFilteredReal)

    dfProbabilities['fakeCount'] = np.log10(dfProbabilities['fakeCount'])
    dfProbabilities['realCount'] = np.log10(dfProbabilities['realCount'])

    return dfProbabilities

def classifyNews(testNews, dfProbabilities):
    resultLabels = ['fF', 'fR', 'rF', 'rR']
    dfResults = pd.DataFrame(0, index=resultLabels, columns=['count'])

    for i in range(len(testNews)):
        news = testNews.iloc[i, 1]
        tag = testNews.iloc[i, 0]

        dfNews = pd.DataFrame(data=news, columns=['word'])
        dfNews = dfNews.merge(dfProbabilities, how='inner',
                              left_on='word', right_on='word')

        fakeScore = dfNews['fakeCount'].sum()
        realScore = dfNews['realCount'].sum()

        prediction = 'F' if fakeScore > realScore else 'R'

        resultKey = tag + prediction
        dfResults.loc[resultKey, 'count'] += 1

    accuracy = round(float(
        (dfResults.loc['fF', 'count'] + dfResults.loc['rR', 'count']) / dfResults['count'].sum()), 4)

    sensitivityReal = round(float(dfResults.loc['rR', 'count'] / (
        dfResults.loc['rR', 'count'] + dfResults.loc['rF', 'count'])), 4)
    sensitivityFake = round(float(dfResults.loc['fF', 'count'] / (
        dfResults.loc['fF', 'count'] + dfResults.loc['fR', 'count'])), 4)

    precisionReal = round(float(dfResults.loc['rR', 'count'] / (
        dfResults.loc['rR', 'count'] + dfResults.loc['fR', 'count'])), 4)
    precisionFake = round(float(dfResults.loc['fF', 'count'] / (
        dfResults.loc['fF', 'count'] + dfResults.loc['rF', 'count'])), 4)

    return accuracy, sensitivityReal, precisionReal, sensitivityFake, precisionFake

def evaluate(dfReal, dfFake, testNews):
    dfMerged = dfReal.merge(dfFake, how='outer')

    countRealUnique = dfMerged.loc[dfMerged['realCount'].isna(
    ), 'word'].count()
    countFakeUnique = dfMerged.loc[dfMerged['fakeCount'].isna(
    ), 'word'].count()

    dfMerged.fillna(1, inplace=True)

    dfMerged.loc[dfMerged['realCount'] > 1, 'realCount'] += 1
    dfMerged.loc[dfMerged['fakeCount'] > 1, 'fakeCount'] += 1

    totalReal = dfMerged['realCount'].sum()
    totalFake = dfMerged['fakeCount'].sum()

    dfMerged['fakeCount'] *= totalReal / totalFake

    # Filtering words with dissimilarity coefficients

    dfMerged['bc'] = abs(dfMerged['realCount'] - dfMerged['fakeCount']) / \
        (dfMerged['realCount'] + dfMerged['fakeCount'])

    # other coefficients could be used here

    # threshold = 0.3
    dfBC = thresholdsCut(dfMerged) # ja vai devolver o dataframe filtrado, precisa filtrar por bc > threshold com varios thresholds e ver qual da melhor acuracia
    #other thresholds could be used here

    # dfMerged = dfMerged.loc[dfMerged['bc'] > 0.3]

    dfProbabilities = probabilitiesCalculation(dfBC, countFakeUnique, countRealUnique)

    # retorna a acuracia, sensibilidade e precisao e f1-score
    return classifyNews(testNews, dfProbabilities)




def calculateStdAndMean(values):
    n = len(values)
    mean = sum(values) / n
    sumSquaredDifferences = sum((x - mean) ** 2 for x in values)
    stdDev = (sumSquaredDifferences / (n - 1)) ** 0.5
    return stdDev, mean


def confidenceInterval(stdDev, mean, n, z=1.96):
    margin = z * stdDev / (n ** 0.5)
    return mean - margin, mean + margin


for i in range(64):
    # voce pode ajustar aqui para fazer o loop para cada coeficiente e cada threshold
    # por enquanto ta fixo, mas tem que fazer o loop para cada coeficiente e cada threshold
    # talvez criar uma funcao que recebe o coeficiente e o threshold e retorna os valores
    # e ai fazer o loop chamando essa funcao
    # ai no final voce tem que ter uma tabela com os intervalos de confiança para cada coeficiente e cada threshold
    
    dfFake = dfFakeOriginal.copy()
    dfReal = dfRealOriginal.copy()


    [dfFakeTest, dfFakeTrain] = splitTrainTest(dfFake)
    [dfRealTest, dfRealTrain] = splitTrainTest(dfReal)
    dfTests = prepareTestDf(dfFakeTest, dfRealTest)
    [dfFakeTrain, dfRealTrain] = prepareTrainDfs(dfFakeTrain, dfRealTrain)

    [accuracy, sensReal, precReal, sensFake, precFake] = evaluate(dfRealTrain, dfFakeTrain, dfTests)

    # aqui vai ser chamada a funcao que calcula o intervalo de confiança
    accuracies.append(accuracy)
    sensitivitiesReal.append(sensReal)
    precisionsReal.append(precReal)
    sensitivitiesFake.append(sensFake)
    precisionsFake.append(precFake)


# calcula o intervalo de confiança para cada coeficiente e cada threshold

stdPrecFake, meanPrecFake = calculateStdAndMean(precisionsFake)
minPrecFake, maxPrecFake = confidenceInterval(
    stdPrecFake, meanPrecFake, len(precisionsFake))

stdPrecReal, meanPrecReal = calculateStdAndMean(precisionsReal)
minPrecReal, maxPrecReal = confidenceInterval(
    stdPrecReal, meanPrecReal, len(precisionsReal))

stdSensReal, meanSensReal = calculateStdAndMean(sensitivitiesReal)
minSensReal, maxSensReal = confidenceInterval(
    stdSensReal, meanSensReal, len(sensitivitiesReal))

stdSensFake, meanSensFake = calculateStdAndMean(sensitivitiesFake)
minSensFake, maxSensFake = confidenceInterval(
    stdSensFake, meanSensFake, len(sensitivitiesFake))

stdAccuracy, meanAccuracy = calculateStdAndMean(accuracies)
minAccuracy, maxAccuracy = confidenceInterval(
    stdAccuracy, meanAccuracy, len(accuracies))

# cria a tabela (TEM QUE FAZER ISSO PARA CADA COEFICIENTE E CADA THRESHOLD)
summaryData = {
    'Metric': ['Precision Fake', 'Precision Real', 'Sensitivity Real', 'Sensitivity Fake', 'Accuracy'],
    'Minimum': [
        minPrecFake,
        minPrecReal,
        minSensReal,
        minSensFake,
        minAccuracy
    ],
    'Maximum': [
        maxPrecFake,
        maxPrecReal,
        maxSensReal,
        maxSensFake,
        maxAccuracy
    ]
}

# salva a tabela em um arquivo csv de cada coeficiente e threshold
dfConfidenceIntervals = pd.DataFrame(summaryData)
dfConfidenceIntervals['Minimum'] = dfConfidenceIntervals['Minimum'].round(4)
dfConfidenceIntervals['Maximum'] = dfConfidenceIntervals['Maximum'].round(4)

print(dfConfidenceIntervals)
