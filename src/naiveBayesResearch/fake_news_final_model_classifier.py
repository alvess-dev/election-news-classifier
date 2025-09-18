import pandas as pd
import random as rd
import numpy as np


def prepareTestDf(dfFake, dfReal):
    data = []

    shuffledDf = pd.concat([dfFake, dfReal], ignore_index=True)
    indices = list(shuffledDf.index)
    rd.shuffle(indices)
    shuffledDf = shuffledDf.iloc[indices].reset_index(drop=True)

    for i in range(len(shuffledDf)):
        news = shuffledDf.iloc[i, 0]
        tag = news[0]
        news = news[1:]
        data.append([tag, news])

    return pd.DataFrame(data, columns=['tag', 'news'])


def prepareTrainingDfs(dfFake, dfReal):
    for i in range(len(dfFake)):
        news = dfFake.iloc[i, 0]
        news.remove('f')

    for i in range(len(dfReal)):
        news = dfReal.iloc[i, 0]
        news.remove('r')

    words = []
    for i in range(len(dfFake)):
        wordList = dfFake.iloc[i, 0]
        for word in wordList:
            words.append(word)
    dfFake = pd.DataFrame({'word': words})
    dfFake = dfFake.groupby('word').size().reset_index(name='fakeOccurrences')

    words = []
    for i in range(len(dfReal)):
        wordList = dfReal.iloc[i, 0]
        for word in wordList:
            words.append(word)
    dfReal = pd.DataFrame({'word': words})
    dfReal = dfReal.groupby('word').size().reset_index(name='realOccurrences')

    return dfFake, dfReal


def evaluateModel(dfReal, dfFake, testNews):

    mergedDf = dfReal.merge(dfFake, how='outer')

    missingInReal = mergedDf.loc[mergedDf['realOccurrences'].isna(
    ), 'word'].count()
    missingInFake = mergedDf.loc[mergedDf['fakeOccurrences'].isna(
    ), 'word'].count()

    mergedDf.fillna(1, inplace=True)

    mergedDf.loc[mergedDf['realOccurrences'] > 1, 'realOccurrences'] += 1
    mergedDf.loc[mergedDf['fakeOccurrences'] > 1, 'fakeOccurrences'] += 1

    totalReal = mergedDf['realOccurrences'].sum()
    totalFake = mergedDf['fakeOccurrences'].sum()

    mergedDf['fakeOccurrences'] *= totalReal / totalFake

    mergedDf['bc'] = abs(mergedDf['realOccurrences'] - mergedDf['fakeOccurrences']) / \
        (mergedDf['realOccurrences'] + mergedDf['fakeOccurrences'])

    mergedDf = mergedDf.loc[mergedDf['bc'] > 0.3]

    totalFilteredFake = mergedDf['fakeOccurrences'].sum()
    totalFilteredReal = mergedDf['realOccurrences'].sum()

    mergedDf['fakeOccurrences'] = mergedDf['fakeOccurrences'] / \
        (missingInFake + totalFilteredFake)
    mergedDf['realOccurrences'] = mergedDf['realOccurrences'] / \
        (missingInReal + totalFilteredReal)

    mergedDf['fakeOccurrences'] = np.log10(mergedDf['fakeOccurrences'])
    mergedDf['realOccurrences'] = np.log10(mergedDf['realOccurrences'])

    resultLabels = ['fF', 'fR', 'rF', 'rR']
    resultCounts = pd.DataFrame(0, index=resultLabels, columns=['count'])

    for i in range(len(testNews)):
        newsWords = testNews.iloc[i, 1]
        trueTag = testNews.iloc[i, 0]

        dfNews = pd.DataFrame(data=newsWords, columns=['word'])

        dfNews = dfNews.merge(
            mergedDf, how='inner', left_on='word', right_on='word')

        scoreFake = dfNews['fakeOccurrences'].sum()
        scoreReal = dfNews['realOccurrences'].sum()

        predictedTag = 'F' if scoreFake > scoreReal else 'R'

        resultKey = trueTag + predictedTag
        resultCounts.loc[resultKey, 'count'] += 1

    accuracy = round(float(
        (resultCounts.loc['fF', 'count'] + resultCounts.loc['rR', 'count']) / resultCounts['count'].sum()), 4)

    sensitivityReal = round(float(
        resultCounts.loc['rR', 'count'] / (resultCounts.loc['rR', 'count'] + resultCounts.loc['rF', 'count'])), 4)
    sensitivityFake = round(float(
        resultCounts.loc['fF', 'count'] / (resultCounts.loc['fF', 'count'] + resultCounts.loc['fR', 'count'])), 4)

    precisionReal = round(float(
        resultCounts.loc['rR', 'count'] / (resultCounts.loc['rR', 'count'] + resultCounts.loc['fR', 'count'])), 4)
    precisionFake = round(float(
        resultCounts.loc['fF', 'count'] / (resultCounts.loc['fF', 'count'] + resultCounts.loc['rF', 'count'])), 4)

    print(resultCounts)
    return accuracy, sensitivityReal, precisionReal, sensitivityFake, precisionFake


dfFakeTrain = pd.read_json('data/train/fakeTrain.json',
                           orient='records', lines=True)
dfRealTrain = pd.read_json('data/train/realTrain.json',
                           orient='records', lines=True)

dfFakeTest = pd.read_json('data/test/fakeTest.json',
                          orient='records', lines=True)
dfRealTest = pd.read_json('data/test/realTest.json',
                          orient='records', lines=True)

dfTestSet = prepareTestDf(dfFakeTest, dfRealTest)
dfFakeTrain, dfRealTrain = prepareTrainingDfs(dfFakeTrain, dfRealTrain)

accuracy, sensitivityReal, precisionReal, sensitivityFake, precisionFake = evaluateModel(
    dfRealTrain, dfFakeTrain, dfTestSet)

metricsData = {
    'Metric': ['Accuracy', 'Sensitivity (Real)', 'Precision (Real)', 'Sensitivity (Fake)', 'Precision (Fake)'],
    'Value': [
        accuracy,
        sensitivityReal,
        precisionReal,
        sensitivityFake,
        precisionFake
    ]
}

dfMetrics = pd.DataFrame(metricsData)

print(dfMetrics)
