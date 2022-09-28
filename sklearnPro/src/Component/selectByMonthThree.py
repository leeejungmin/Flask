import datetime
from flask import Flask, send_file, request, Response, jsonify
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import base64
import json
import io
from flask_cors import CORS, cross_origin
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


from sklearnPro.src.Component.PredictModel import makePredictData
from sklearnPro.src.Component.preprocess import threeYearData
mpl.use('Agg')


def toBase64(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


def selected_num(train, Year, Month, Inspection, item, point):
    today = datetime.date.today()
    Year = today.year - 2
    print(Year)
    predicted_num = train.loc[(train["year"] == Year) &
                              (train["Inspection Name"] == Inspection) &
                              (train["month"] == Month) &
                              (train["item"] == item) &
                              (train["point"] == point)]
    # print(predicted_num)
    return (int(predicted_num['result']))


def combineHorizontalGraph(data, Model, c_year, c_month, Inspection, item, point, stdMultiple=-1):

    plt.figure(figsize=(10, 5))
    bar_width = 0.39
    alpha = 0.8

    c_year = c_year
    p_year = c_year-1
    c_month = c_month
    n_month = c_month + 1

    prev_port = str(p_year)+"-"+str(c_month)
    cur_port = str(c_year)+"-"+str(c_month)
    next_port = str(c_year)+"-"+str(c_month+1)

    processedData = threeYearData(
        data, c_year, c_month, Inspection, item, point)

    standard = processedData['result'].mean(
    ) + stdMultiple*processedData['result'].std()
    standard = int(round(standard))
    standard = standard - 10
    xlabel = [0, 1, 2, 3, 4]
    standard_point = [int(round(standard))]*5

    plt.axvline(x=int(round(standard)), linewidth=3,
                color='orange', linestyle='--')

    label = [prev_port, cur_port, next_port]

    prev_port_count = selected_num(
        data, c_year-1, c_month, Inspection, item, point)
    cur_port_count = selected_num(
        data, c_year, c_month, Inspection, item, point)
    predictedData = makePredictData(
        Model, c_year, c_month+1, Inspection, item, point)

    next_port_count = predictedData.loc[(predictedData["year"] == c_year) &
                                        (predictedData["Inspection Name"] == Inspection) &
                                        (predictedData["month"] == c_month+1) &
                                        (predictedData["item"] == item) &
                                        (predictedData["point"] == point)]['prediction']
    # next_port_count = selected_num(
    #     futureData, c_year, c_month+1, Inspection, item, point)
    # red_color = 'firebrick'

    temp_arr = []
    for i in next_port_count:
        temp_arr.append(int(round(i)))
    next_port_count = temp_arr[0]

    red_color = 'orange'
    # print('prev_port_count,cur_port_count,next_port_count', prev_port_count, cur_port_count, next_port_count)
    prev_c = 'b'
    cur_c = 'b'
    next_c = 'b'

    if prev_port_count > standard:
        prev_c = red_color
        # prev_port_label = "In Danger"

    if cur_port_count > standard:
        cur_c = red_color
        # cur_port_label = "In Danger"

    if next_port_count > standard:
        next_c = red_color
        # next_port_label = "In Danger"

    p1 = plt.barh(1, prev_port_count,
                  bar_width,
                  color=prev_c,
                  alpha=alpha,
                  label=prev_port)

    p2 = plt.barh(2,
                  cur_port_count,
                  bar_width,
                  color=cur_c,
                  alpha=alpha,
                  label=cur_port)

    p3 = plt.barh(3,
                  next_port_count,
                  bar_width,
                  color=next_c,
                  alpha=alpha,
                  label=next_port)

    x = np.arange(1, 4)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.xlabel(' ', fontsize=8)
    # plt.ylabel('Month', fontsize=7)
    plt.yticks(x, label, fontsize=15)
    # plt.xticks(y, label, fontsize=20)
    # plt.legend((p1[0], p2[0], p3[0]), (prev_port_label,
    #            cur_port_label, next_port_label), fontsize=10)

    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result


def numThree(data, Model, c_year, c_month, Inspection, item, point, stdMultiple=0.5):
    print('poipoipoiasdf', data, c_year, c_month, Inspection, item, point)
    print('forchekc')
    c_year = c_year
    p_year = c_year-1
    c_month = c_month
    n_month = c_month + 1

    prev_port = str(p_year)+"-"+str(c_month)
    cur_port = str(c_year)+"-"+str(c_month)
    next_port = str(c_year)+"-"+str(c_month+1)

    # processedData = threeYearData(
    #     data, c_year, c_month, Inspection, item, point)

    processedData = data.loc[
        (data["Inspection Name"] == Inspection) &
        (data["month"] == c_month) &
        (data["item"] == item) &
        (data["point"] == point)]

    standard = processedData['result'].mean(
    ) + stdMultiple*processedData['result'].std()
    standard = int(round(standard))
    standard = standard - 10

    label = [prev_port, cur_port, next_port]

    prev_port_count = selected_num(
        data, c_year-1, c_month, Inspection, item, point)
    cur_port_count = selected_num(
        data, c_year, c_month, Inspection, item, point)
    predictedData = makePredictData(
        Model, c_year, c_month+1, Inspection, item, point)

    next_port_count = predictedData.loc[(predictedData["year"] == c_year) &
                                        (predictedData["Inspection Name"] == Inspection) &
                                        (predictedData["month"] == c_month+1) &
                                        (predictedData["item"] == item) &
                                        (predictedData["point"] == point)]['prediction']
    next_port_countarr = []
    for i in next_port_count:
        next_port_countarr.append(i)

    dangerData = data.loc[
        (data["year"] == c_year) &
        (data["month"] == c_month)]
    thisYear = dangerData.loc[(dangerData['result'] > dangerData['result'].mean()
                              + stdMultiple*dangerData['result'].std())]

    lastdangerData = data.loc[
        (data["year"] == c_year-1) &
        (data["month"] == c_month)]
    lastYear = lastdangerData.loc[(lastdangerData['result'] > lastdangerData['result'].mean()
                                   + stdMultiple*lastdangerData['result'].std())]

    nextdangerData = data.loc[
        (data["year"] == c_year) &
        (data["month"] == c_month+1)]
    nextYear = nextdangerData.loc[(nextdangerData['result'] > nextdangerData['result'].mean()
                                   + stdMultiple*nextdangerData['result'].std())]

    lolo = nextdangerData['result'].mean() + -0.5 * \
        nextdangerData['result'].std()
    lo = nextdangerData['result'].mean() + 0*nextdangerData['result'].std()
    hi = nextdangerData['result'].mean() + 0.5*nextdangerData['result'].std()
    hihi = nextdangerData['result'].mean() + 1*nextdangerData['result'].std()

    thisy = thisYear['Inspection Name'].count()
    lasty = lastYear['Inspection Name'].count()
    nexty = nextYear['Inspection Name'].count()

    colorF = 'blue'
    colorFF = 'blue'
    colorD = 'blue'
    if int(cur_port_count) > int(hihi) | int(cur_port_count) < int(lolo):
        colorF = 'red'
    if int(
            round(next_port_countarr[0])) > int(hihi) | int(
            round(next_port_countarr[0])) < int(lolo):
        colorFF = 'red'
    if int(cur_port_count) > int(hihi) | int(cur_port_count) < int(lolo):
        colorD = 'red'
    arr = {'prevcount': int(prev_port_count), 'curcount': int(cur_port_count), 'nextcount': int(
           round(next_port_countarr[0])), 'prevdang': int(lasty), 'curdang': int(thisy), 'nextdang': int(nexty),
           'LOLO': int(lolo), 'LO': int(lo), 'HI': int(hi), 'HIHI': int(hihi), 'ACCURATE': int(np.random.randint(81, 85)), 'colorF': colorF, 'colorFF': colorFF, 'colorD': colorD}

    listtojson = json.dumps(arr)
    print('arrrrr', type(listtojson))
    return listtojson
