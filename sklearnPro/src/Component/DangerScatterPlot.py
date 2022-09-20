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

from sklearnPro.src.Component.preprocess import nameCategory, toBase64


mpl.use('Qt5Agg')


def findDangerbyInpection(test, Inspection, year, month, stdMultiple):
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


def findDangerbyMonth(test, year, month, stdMultiple):
    processedData = test.loc[(test["year"] == year) & test["month"]]
    DangeredAllData = processedData.loc[(processedData['result'] > processedData['result'].mean()
                                         + stdMultiple*processedData['result'].std())]
    return DangeredAllData
    # 분기 미래예측


def DangerScatterChartByInspection(test, Inspection, year, month, stdMultiple):
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
    data = findDangerbyMonthPeriod(
        test, Inspection, year, month, stdMultiple, typePeriod)
    print(data)
    g = sns.scatterplot(
        data=nameCategory(data[0]), x='year_month', y='point', hue='item', s=50, alpha=0.8
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


def DangerScatterChartByMonth(test, year, month, stdMultiple):
    data = findDangerbyMonth(test, year, month, stdMultiple)

    sns.scatterplot(
        data=nameCategory(data), x='item', y='point', hue='Inspection Name', s=399,
        sizes=(20, 5),  legend='full'
    )
    plt.xticks(rotation=0, fontsize=13)
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
    sns.set(rc={'figure.figsize': (4.5, 3)})
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


def rankFiveDangerInspection(test, year, month, stdMultiple, typePeriod):
    DangeredData = test.loc[(test['result'] > test['result'].mean(
    ) + stdMultiple*test['result'].std())]

#     DangeredData = DangeredAllData.loc[(
#         DangeredAllData['Inspection Name'] == Inspection)]

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

    find5list_corr = json.loads(
        DangeredData['Inspection Name'].value_counts().to_json())

    return find5list_corr



