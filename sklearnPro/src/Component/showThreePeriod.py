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

from sklearnPro.src.Component.PredictModel import makePredictData
from sklearnPro.src.Component.corMatt import toBase64
mpl.use('Agg')


def lineWithBarplotWithYear(train, Model, year, Inspection, item, point, month=1):

    lastyearData = train.loc[(train["year"] == year-1) &
                             (train["Inspection Name"] == Inspection) &
                             (train["item"] == item) &
                             (train["point"] == point)]
                             
    print('lastyearData',lastyearData)
    currentyearData = train.loc[(train["year"] == year) &
                                (train["Inspection Name"] == Inspection) &
                                (train["item"] == item) &
                                (train["point"] == point)]

    futureData = makePredictData(Model, year, month, Inspection, item, point)

    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=futureData, x='month', y='prediction', marker='o', linewidth=5,
                 estimator=None,   ax=ax1,  color='orange', alpha=0.9)

    ax2 = ax1.twinx()

    sns.lineplot(data=lastyearData, x='month', y='result', marker='o', linewidth=2,
                 estimator=None,  ax=ax2,  color='blue', alpha=0.55)

    ax3 = ax2.twinx()

    sns.lineplot(data=currentyearData, x='month', y='result',
                 linewidth=7, ax=ax3, color='blue', alpha=0.8)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # sns.lineplot(data=futureData, x='month', y='prediction', marker='o', linewidth=5,
    #              estimator=None, ax=ax, color='orange', alpha=0.7)
    # sns.lineplot(data=lastyearData, x='month', y='result', marker='o', linewidth=5,
    #              estimator=None, ax=ax,  color='blue', alpha=0.7)
    # sns.barplot(data=currentyearData, x='month', y='result', marker='o', linewidth=5,
    #             estimator=None, ax=ax,  color='blue', alpha=0.7)

    sns.despine(top=True, right=True, left=True,
                bottom=True, offset=True, trim=True)

    ax1.tick_params(top=False, labeltop=False, left=False, labelleft=True,
                    right=False, labelright=False, bottom=False, labelbottom=True)
    ax2.tick_params(top=False, labeltop=False, left=False, labelleft=False,
                    right=False, labelright=False, bottom=False, labelbottom=True)
    ax3.tick_params(top=False, labeltop=False, left=False, labelleft=False,
                    right=False, labelright=False, bottom=False, labelbottom=True)

    result = toBase64(fig)

    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    plt.clf()
    plt.close()

    return result
