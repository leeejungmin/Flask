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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# 이번달 목표 lo, hi, hihi
def thisMonthCount(test, Inspection, year, month):
    processedData = test.loc[(test["year"] == year) & (test["month"] == month)]

    LoData = processedData.loc[(processedData['result'] < -processedData['result'].mean(
    ) - processedData['result'].std())]
    HiData = processedData.loc[(processedData['result'] > processedData['result'].mean() + 1.5*processedData['result'].std()) &
                               (processedData['result'] < processedData['result'].mean() + processedData['result'].std())]
    HiHiData = processedData.loc[(processedData['result'] > processedData['result'].mean(
    ) + 1.5*processedData['result'].std())]

    total = processedData['result'].count()
    Lototal = LoData['result'].count()
    HItotal = HiData['result'].count()
    HIHItotal = HiHiData['result'].count()

    Sequence = 'total,Lototal,HItotal,HIHItotal'
    Lotarget = 'blue'
    HItarget = 'blue'
    HIHItarget = 'blue'

    if(Lototal < total*0.7):
        Lotarget = 'red'
    if(HItotal > total*0.2):
        HItarget = 'red'
    if(HIHItotal > total*0.1):
        HIHItarget = 'red'

    target = [Sequence, Lotarget, HItarget, HIHItarget]
    total = [total, Lototal, HItotal, HIHItotal]

    combineArray = [target, total]
    arrinspection = np.array(combineArray)
    find5list_Inspection = json.dumps(arrinspection, cls=NpEncoder)

    return find5list_Inspection


def thisMonthRatio(test, year, month, stdMultiple):
    processedData = test.loc[(test['year'] == year) & (test['month'] == month)]

    processedData = nameCategory(processedData)
    if(stdMultiple == -1):
        Data = processedData.loc[(processedData['result'] < -processedData['result'].mean(
        ) - processedData['result'].std())]
    elif(stdMultiple == 0):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean() - processedData['result'].std()) &
                                 (processedData['result'] < processedData['result'].mean() + 0*processedData['result'].std())]
    elif(stdMultiple == 1):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean() + 0*processedData['result'].std()) &
                                 (processedData['result'] < processedData['result'].mean() + 1*processedData['result'].std())]
    elif(stdMultiple == 1.5):
        Data = processedData.loc[(processedData['result'] > processedData['result'].mean(
        ) + 1.5*processedData['result'].std())]

    n = Data['Inspection Name'].drop_duplicates().count()

    arrInspectName = []
    arrInspectCount = []

    for x in Data['Inspection Name'].drop_duplicates():
        arrInspectName.append(x)

    for inspection in arrInspectName:
        countData = Data.loc[(Data['Inspection Name'] == inspection)]
        arrInspectCount.append(countData['Inspection Name'].count())

    explode = [0.05] * n
    colors = sns.color_palette('pastel')[:n]

    Data.sort_values("Inspection Name", inplace=True)
    plt.pie(arrInspectCount, labels=arrInspectName,
            counterclock=False,  colors=colors, autopct='%.0f %%', textprops={'fontsize': 23})

    result = toBase64(plt)
    plt.clf()
    plt.close()

    return result
