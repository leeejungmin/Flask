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
import time

mpl.use('Agg')


def toBase64(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


def normal_dis(train, Inspection,  point):
    train = train.loc[(train["Inspection Name"] == Inspection)]

    ncols = len(train["point"].value_counts().keys())

    fig, axes = plt.subplots(ncols=ncols, nrows=1)
    fig.set_size_inches(5*ncols, 5)

    predicted_array = [[] for i in range(ncols)]
    predicted_num_dis = [[] for i in range(ncols)]

    for idx, point_idx in enumerate(train["point"].value_counts().keys()):
        # print(idx, point_idx, train["point"].value_counts().keys())
        predicted_array[idx] = train.loc[
            (train["Inspection Name"] == Inspection) &
            (train["point"] == point_idx)
        ]
        predicted_num_dis[idx] = sns.distplot(
            predicted_array[idx]["result"], ax=axes[idx])

        predicted_num_dis[idx].spines['top'].set_visible(False)
        predicted_num_dis[idx].spines['right'].set_visible(False)
        predicted_num_dis[idx].spines['left'].set_visible(False)
        predicted_num_dis[idx].spines['bottom'].set_visible(False)

        predicted_num_dis[idx].set(ylabel=None)
        predicted_num_dis[idx].set(xlabel=None)

        title_mapping = {"Hydraulic System Check": 1, "Camshaft": 2, "Cylinder": 3, "Engine Check": 4,
                         "Table Roller": 5, "Reducer Check": 6, "FuelRail": 7, "InterCooler": 8, "Geared Motor Check": 9}

        point_mapping = {'coupling': 1, 'oil': 2, 'tank': 3, ' in the filter': 4, 'fuelRail': 5, 'bearing': 6,
                         'same position(vertical)': 7, 'bearing cover': 8, 'gear': 9, 'coolant': 10, 'cooling fan': 11, 'roller': 12, 'in the filter': 13, 'pump': 14, 'o-ring': 15}
        item_mapping = {'alignment': 1, 'vibration': 2, 'temperature': 3,
                        'intensity': 4, 'viscosity': 5, 'O2 saturation': 6, 'CO2 saturation': 7}

        reversed_title_mapping = dict((v, k) for k, v in title_mapping.items())
        reversed_point_mapping = dict((v, k) for k, v in point_mapping.items())
        reversed_item_mapping = dict((v, k) for k, v in item_mapping.items())

        hfont = {'fontname': 'Arial'}
        axes[idx].set_xlabel(
            reversed_item_mapping[point_idx], fontsize=15, **hfont)

    result = toBase64(fig)

    plt.clf()
    plt.close()

    return result


def normal_dis_avant(train, Inspection, item, point):
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(ncols=4, nrows=1)
    fig.set_size_inches(20, 5)

    train = train.loc[(train["Inspection Name"] == 1)]
    print('train["point"].value_counts()', train["point"].value_counts())

    predicted_num1 = train.loc[
        (train["Inspection Name"] == Inspection) &
        (train["point"] == 1)
    ]

    predicted_num2 = train.loc[
        (train["Inspection Name"] == Inspection) &
        (train["point"] == 2)
    ]
    predicted_num3 = train.loc[
        (train["Inspection Name"] == Inspection) &
        (train["point"] == 3)
    ]
    predicted_num4 = train.loc[
        (train["Inspection Name"] == Inspection) &
        (train["point"] == 4)
    ]

    predicted_num0_dis = sns.distplot(predicted_num1["result"], ax=ax1)
    predicted_num1_dis = sns.distplot(
        predicted_num2["result"], ax=ax2)
    predicted_num2_dis = sns.distplot(predicted_num3["result"], ax=ax3)
    predicted_num3_dis = sns.distplot(predicted_num4["result"], ax=ax4)

    predicted_num1_dis.spines['top'].set_visible(False)
    predicted_num1_dis.spines['right'].set_visible(False)
    predicted_num1_dis.spines['left'].set_visible(False)
    predicted_num1_dis.spines['bottom'].set_visible(False)

    predicted_num2_dis.spines['top'].set_visible(False)
    predicted_num2_dis.spines['right'].set_visible(False)
    predicted_num2_dis.spines['left'].set_visible(False)
    predicted_num2_dis.spines['bottom'].set_visible(False)

    predicted_num3_dis.spines['top'].set_visible(False)
    predicted_num3_dis.spines['right'].set_visible(False)
    predicted_num3_dis.spines['left'].set_visible(False)
    predicted_num3_dis.spines['bottom'].set_visible(False)

    predicted_num0_dis.spines['top'].set_visible(False)
    predicted_num0_dis.spines['right'].set_visible(False)
    predicted_num0_dis.spines['left'].set_visible(False)
    predicted_num0_dis.spines['bottom'].set_visible(False)
    # ax2.set_xlabel(,fontsize=20)
    # predicted_numg1.set(yticklabels=[])
    predicted_num1_dis.set(ylabel=None)
    predicted_num2_dis.set(ylabel=None)
    predicted_num3_dis.set(ylabel=None)

    result = toBase64(fig)

    plt.clf()
    plt.close()

    return result
