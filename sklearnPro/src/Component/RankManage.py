from datetime import datetime
from flask import Flask, send_file, request, Response, jsonify
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import time
import json
import re

from sklearnPro.src.Component.preprocess import nameCategory, toBase64


mpl.use('Qt5Agg')


def topRankBarplot(test, typeName, year, stdMultiple):

    processedData = test.loc[(test['year'] == year)]

    processedData = nameCategory(processedData)
    if(stdMultiple == 1):
        Data = processedData.loc[(processedData['result'] < processedData['result'].mean(
        ) + processedData['result'].std())]
    elif(stdMultiple == 1.5):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean() * processedData['result'].std()) &
                                 (processedData['result'] < processedData['result'].mean() + 1.5*processedData['result'].std())]
    elif(stdMultiple == 2):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean() + 2*processedData['result'].std()) &
                                 (processedData['result'] < processedData['result'].mean() + 3*processedData['result'].std())]
    elif(stdMultiple == 3):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean(
        ) + 3*processedData['result'].std())]

    pointTopListkeys = Data.groupby(typeName).count(
    )['Inspection Name'].sort_values(ascending=False)[:3].keys()
    pointTopListValues = Data.groupby(typeName).count(
    )['Inspection Name'].sort_values(ascending=False)[:3]
    pointTopListValuesArr = []
    for value in pointTopListValues:
        pointTopListValuesArr.append(value)

    # inspection name 배열 top 3, point top3, item top3 , 이름, count

    plt.figure(figsize=(10, 5))
    bar_width = 0.39
    alpha = 0.7

    label = pointTopListkeys

    prev_c = 'b'
    cur_c = 'b'
    next_c = 'b'

    p1 = plt.barh(1, pointTopListValuesArr[2],
                  bar_width,
                  color=prev_c,
                  alpha=alpha,
                  label=pointTopListkeys[2])

    p2 = plt.barh(2,
                  pointTopListValuesArr[1],
                  bar_width,
                  color=cur_c,
                  alpha=alpha,
                  label=pointTopListkeys[1])

    p3 = plt.barh(3,
                  pointTopListValuesArr[0],
                  bar_width,
                  color=next_c,
                  alpha=alpha,
                  label=pointTopListkeys[0])

    x = np.arange(1, len(pointTopListkeys)+1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.gca().spines['bottom'].set_visible(False)

    plt.xlabel(' ', fontsize=8)

    plt.yticks(x, pointTopListkeys, fontsize=15)

    result = toBase64(plt)
    plt.clf()
    plt.close()

    return result
