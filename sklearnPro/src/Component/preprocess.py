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
from datetime import datetime


def toBase64(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


def nameCategory(data):
    title_mapping = {'Hydraulic System': 1, 'Camshaft': 2, 'Cylinder': 3, 'Engine': 4,
                     'Table Roller': 5, 'Reducer': 6, ' FuelRail': 7, 'InterCooler': 8, 'Geared Motor': 9}
    point_mapping = {'coupling': 1, 'oil': 2, 'tank': 3, 'filter(in)': 4, 'fuelRail': 5, 'bearing': 6,
                     'position(ver)': 7, 'cover': 8, 'gear': 9, 'coolant': 10, 'coolingfan': 11, 'roller': 12, 'pump': 14, 'o-ring': 15}
    item_mapping = {'alignment': 1, 'vibration': 2, 'temperature': 3,
                    'intensity': 4, 'viscosity': 5, 'O2 saturation': 6, 'CO2 saturation': 7}
    season_mapping = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
    quater_mapping = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}

    reversed_title_mapping = dict((v, k) for k, v in title_mapping.items())
    reversed_point_mapping = dict((v, k) for k, v in point_mapping.items())
    reversed_item_mapping = dict((v, k) for k, v in item_mapping.items())
    reversed_quarter_mapping = dict((v, k) for k, v in quater_mapping.items())

    data['Inspection Name'] = data['Inspection Name'].map(
        reversed_title_mapping)
    data['point'] = data['point'].map(reversed_point_mapping)
    data['item'] = data['item'].map(reversed_item_mapping)
    data['quarter'] = data['quarter'].map(reversed_quarter_mapping)

    return data


def selected_num(train, Year, Month, Inspection, item, point):
    predicted_num = train.loc[(train["year"] == Year) &
                              (train["Inspection Name"] == Inspection) &
                              (train["month"] == Month) &
                              (train["item"] == item) &
                              (train["point"] == point)]
    return (int(predicted_num['result']))


def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)


def threeYearData(train, Year, Month, Inspection, item, point):
    endDate = '01/0' + str(Month) + '/' + str(Year)
    startDate = '31/0' + str(Month) + '/' + str(Year-3)

    if(len(str(Month)) == 2):
        endDate = '01/' + str(Month) + '/' + str(Year)
        startDate = '31/' + str(Month) + '/' + str(Year-3)

    datetime_object_startDate = datetime.strptime(startDate, "%d/%m/%Y")
    datetime_object_endDate = datetime.strptime(endDate, '%d/%m/%Y')
    # train = train.date_range(datetime_object_startDate,
    #                          datetime_object_endDate)
    train = train.loc[(train['date'] <= datetime_object_endDate)
                      & (train['date'] >= datetime_object_startDate)]

    predicted_num = train.loc[(train["Inspection Name"] == Inspection) &
                              (train["item"] == item) &
                              (train["point"] == point)]

    return predicted_num


def nameCategory(data):

    data2 = data.copy()
    title_mapping = {'HydraulicSystem': 1, 'Camshaft': 2, 'Cylinder': 3, 'Engine': 4,
                     'TableRoller': 5, 'Reducer': 6, ' FuelRail': 7, 'InterCooler': 8, 'GearedMotor': 9}
    point_mapping = {'coupling': 1, 'oil': 2, 'tank': 3, 'filter(in)': 4, 'fuelRail': 5, 'bearing': 6,
                     'position(ver)': 7, 'cover': 8, 'gear': 9, 'coolant': 10, 'cooling-fan': 11, 'roller': 12, 'filter(in)': 13, 'pump': 14, 'o-ring': 15}
    item_mapping = {'alignment': 1, 'vibration': 2, 'temperature': 3,
                    'intensity': 4, 'viscosity': 5, 'O2 saturation': 6, 'CO2 saturation': 7}
    season_mapping = {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3}
    quater_mapping = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}

    reversed_title_mapping = dict((v, k) for k, v in title_mapping.items())
    reversed_point_mapping = dict((v, k) for k, v in point_mapping.items())
    reversed_item_mapping = dict((v, k) for k, v in item_mapping.items())
    reversed_quarter_mapping = dict((v, k) for k, v in quater_mapping.items())

    data2['Inspection Name'] = data2['Inspection Name'].map(
        reversed_title_mapping)
    data2['point'] = data2['point'].map(reversed_point_mapping)
    data2['item'] = data2['item'].map(reversed_item_mapping)
    data2['quarter'] = data2['quarter'].map(reversed_quarter_mapping)

    return data2


def ChildListOfFacility(test, Inspection):

    processedData = test.loc[(test["Inspection Name"] == Inspection)]
#     processedData = processedData.drop_duplicates(subset=["point", "item"], keep=False)

    processedData = nameCategory(processedData)
    listOfPoint = processedData["point"].drop_duplicates()
    listOfItem = processedData["item"].drop_duplicates()

    arrPoint = []
    arrItem = []
    for i in listOfPoint:
        if(i == 'NaN'):
            pass
        arrPoint.append(i)

    for i in listOfItem:
        if(i == 'NaN'):
            pass
        arrItem.append(i)

    combine = arrPoint, arrItem
    jsonString = json.dumps(combine)

    print(type(jsonString))
    return jsonString
