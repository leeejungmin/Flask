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
from sklearnPro.src.Component.PredictModel import makePredictData, predictModel

from sklearnPro.src.Component.preprocess import nameCategory, toBase64


mpl.use('Qt5Agg')

stdMultiple = 0


def findDangerbyInpection(test, Inspection, year, month, stdMultiple):
    stdMultiple = 0

    processedData = test.loc[(test[year] == year)],
    DangeredAllData = processedData.loc[(processedData['result'] > processedData['result'].mean(
    ) + stdMultiple*processedData['result'].std())],

    # 미래 위험 종목 예측,
    DangeredData = DangeredAllData.loc[(
        DangeredAllData['Inspection Name'] == Inspection)]
    endmonth = month + 4,
    if endmonth > 13:
        endmonth = endmonth - 12
    FutureDangeredData = DangeredAllData.loc[(
        DangeredAllData['month'] < endmonth) & (DangeredAllData['month'] > month)]

    return FutureDangeredData


def DangerScatterChartByInspection(test, Inspection, year, month, stdMultiple):

    stdMultiple = 0

    data = findDangerbyInpection(test, Inspection, year, month, stdMultiple)
    print(data)
    sns.scatterplot(
        data=nameCategory(data), x='point', y='item', hue='Inspection Name', s=200,
        sizes=(20, 5),  legend='full'
    )
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.set_context("paper", rc={"font.size": 18,
                                 "axes.titlesize": 18, "axes.labelsize": 18})

    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result


def findDangerbyMonthPeriod(test, Inspection, year, month, stdMultiple, typePeriod):

    stdMultiple = 0

    DangeredAllData = test.loc[(test['result'] > test['result'].mean(
    ) + stdMultiple*test['result'].std())]

    DangeredData = DangeredAllData.loc[(
        DangeredAllData['Inspection Name'] == Inspection)]

    if typePeriod == 'Y':
        period = 12
    elif typePeriod == 'H':
        period = 6
    elif typePeriod == 'Q':
        period = 4
    elif typePeriod == 'M':
        period = 1

    endYear = year
    endmonth = month + period

    if endmonth > 12:
        endmonth = endmonth - 12
        endYear = endYear + 1

        tempmonth = month
        month = endmonth
        endmonth = tempmonth

    startDate = '01/0' + str(month) + '/' + str(year)
    endDate = '31/0' + str(endmonth) + '/' + str(endYear)
    if(len(str(month)) == 2):
        startDate = '01/' + str(month) + '/' + str(year)
        endDate = '31/' + str(endmonth) + '/' + str(endYear)

    datetime_object_startDate = datetime.strptime(startDate, "%d/%m/%Y")
    datetime_object_endDate = datetime.strptime(endDate, '%d/%m/%Y')

    FutureDangeredData = DangeredAllData.loc[(DangeredAllData['date'] <= datetime_object_endDate)
                                             & (DangeredAllData['date'] >= datetime_object_startDate)]

    return FutureDangeredData, month, endmonth


def DangerScatterChartByMonthPeriod(test, Inspection, year, month, stdMultiple, typePeriod):

    stdMultiple = 0

    data = findDangerbyMonthPeriod(
        test, Inspection, year, month, stdMultiple, typePeriod)

    g = sns.scatterplot(
        data=nameCategory(data[0]), x='year_month', y='point', hue='item', s=50, alpha=0.9
    )

    plt.xticks(rotation=30, fontsize=5.5)
    plt.yticks(rotation=0, fontsize=6.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # sns.set_context("paper", rc={"font.size": 4,
    #                              "axes.titlesize": 4, "axes.labelsize": 4})

    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    # sns.set(rc={'figure.figsize': (5, 3)})

    # _legend.remove()
    # plt.legend(legend_elements("sizes", num=5))
    plt.legend(loc="lower left",  fontsize=5)

    result = toBase64(plt)
    plt.clf()
    plt.close()

    return result


def findDangerbyMonth(test, year, month, stdMultiple):
    stdMultiple = 0

    processedData = test.loc[(test["year"] == year) & test["month"] == month]
    # DangeredAllData = processedData.loc[(processedData['result'] > processedData['result'].mean()
    #                                      + stdMultiple*processedData['result'].std())]
    return processedData
    # 분기 미래예측


def DangerScatterChartByMonth(test, year, month, stdMultiple):
    # data = findDangerbyMonth(test, year, month, stdMultiple)

    processedData = test.loc[
        (test["year"] == year) &
        (test["month"] == month)]
        
    if(stdMultiple==1.5):
        data = processedData.loc[(processedData['result'] > processedData['result'].mean()
                                + stdMultiple*processedData['result'].std())]
    if(stdMultiple==1):
        data = processedData.loc[(processedData['result'] > processedData['result'].mean()
                                + stdMultiple*processedData['result'].std())]
    if(stdMultiple==0):
        data = processedData.loc[(processedData['result'] > processedData['result'].mean()
                                + stdMultiple*processedData['result'].std())]
    if(stdMultiple==-1):
        data = processedData.loc[(processedData['result'] < processedData['result'].mean()
                                + stdMultiple*processedData['result'].std())]
    print('ScatterMonth...........', data)
    sns.scatterplot(
        data=nameCategory(data), x='item', y='point', hue='Inspection Name', s=399,
        sizes=(3, 5),  legend='full'
    )
    plt.xticks(rotation=40, fontsize=13)
    plt.yticks(rotation=40, fontsize=13)
    # plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.legend(fontsize='13')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.set_context("paper", rc={"font.size": 17,
                                 "axes.titlesize": 17, "axes.labelsize": 17})
    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    # sns.set(rc={'figure.figsize': (3, 5)})
    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result


def DangerScatterChartByPredict(test, year, month, stdMultiple):
    stdMultiple = 0
    print('predict....',test,year,month,stdMultiple);
    processedData = test.loc[
        (test["year"] == year) &
        (test["month"] == month+1)]
    data = processedData.loc[(processedData['result'] > processedData['result'].mean()
                              + stdMultiple*processedData['result'].std())]
    sns.scatterplot(
        data=nameCategory(data), x='item', y='point', hue='Inspection Name', s=399,
        sizes=(3, 5),  legend='full'
    )
    plt.xticks(rotation=40, fontsize=13)
    plt.yticks(rotation=40, fontsize=13)
    # plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.legend(fontsize='13')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.set_context("paper", rc={"font.size": 17,
                                 "axes.titlesize": 17, "axes.labelsize": 17})
    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    # sns.set(rc={'figure.figsize': (3, 5)})
    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result


def findDangerbyQuater(test, year, stdMultiple):
    processedData = test.loc[(test["year"] == year)]
    DangeredAllData = processedData.loc[(processedData['result'] > processedData['result'].mean()
                                         + stdMultiple*processedData['result'].std())]
    return DangeredAllData


def DangerScatterChartByQuater(test, year, stdMultiple):
    data = findDangerbyQuater(test, year, stdMultiple)
    sns.scatterplot(
        data=nameCategory(data), x='quarter', y='point', hue='Inspection Name', size='result',
        sizes=(20, 200)
    )
    plt.xticks(rotation=20, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    sns.set_context("paper", rc={"font.size": 18,
                                 "axes.titlesize": 18, "axes.labelsize": 18})
    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    sns.set(rc={'figure.figsize': (7, 9)})

    plt.legend(loc='lower right', fontsize=10,
               bbox_to_anchor=(1.45, 0),
               title="Delivery Type",
               title_fontsize=12,
               facecolor='white')
    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result


def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def rankFiveDangerInspection(test, year, month, stdMultiple, typePeriod):
    DangeredData = test.loc[(test['result'] > test['result'].mean(
    ) + stdMultiple*test['result'].std())]

    DangeredData = nameCategory(DangeredData)

    if typePeriod == '3Y':
        period = 36
    elif typePeriod == '2Y':
        period = 24
    elif typePeriod == 'Y':
        period = 12
    elif typePeriod == 'H':
        period = 6
    elif typePeriod == 'Q':
        period = 4
    elif typePeriod == 'M':
        period = 1

    endYear = year
    endmonth = month + period

    if endmonth > 12:
        endmonth = endmonth - 12
        endYear = endYear + 1

        tempmonth = month
        month = endmonth
        endmonth = tempmonth

    startDate = '01/0' + str(month) + '/' + str(year)
    endDate = '31/0' + str(endmonth) + '/' + str(endYear)

    if(len(str(month)) == 2):
        startDate = '01/' + str(month) + '/' + str(year)
        endDate = '31/' + str(endmonth) + '/' + str(endYear)

    find5list_inspection = json.loads(
        DangeredData['Inspection Name'].value_counts().to_json())

    find5list_point = json.loads(
        DangeredData['point'].value_counts().to_json())

    find5list_item = json.loads(
        DangeredData['item'].value_counts().to_json())

    Dangerinspection = find5list_inspection.items()
    listinspection = list(Dangerinspection)
    arrinspection = np.array(listinspection)

    Dangerpoint = find5list_point.items()
    listpoint = list(Dangerpoint)
    arrpoint = np.array(listpoint)

    Dangeritem = find5list_item.items()
    listitem = list(Dangeritem)
    arritem = np.array(listitem)

    find5list_Inspection = json.dumps(arrinspection,
                                      cls=NumpyEncoder)

    find5list_Point = json.dumps(arrpoint,
                                 cls=NumpyEncoder)

    find5list_Item = json.dumps(arritem,
                                cls=NumpyEncoder)

    return find5list_Inspection, find5list_Point, find5list_Item
