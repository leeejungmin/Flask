from datetime import datetime
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

from sklearnPro.src.Component.baseGraph import toBase64
from sklearnPro.src.Component.preprocess import nameCategory
mpl.use('Agg')


def findYearByFactor(train, Inspection, Year, Month, item, point, typeHue):
    processedData = train.loc[
        (train["Inspection Name"] == Inspection) &
        # (train["month"] == Month) &
        (train["item"] == item) &
        (train["point"] == point)]
    # endDate = '01/0' + str(Month+1) + '/' + str(Year)
    # startDate = '31/0' + str(Month) + '/' + str(Year-1)

    # if(len(str(Month)) == 2):
    #     endDate = '01/' + str(Month+1) + '/' + str(Year)
    #     startDate = '31/' + str(Month) + '/' + str(Year-1)

    # datetime_object_startDate = datetime.strptime(startDate, "%d/%m/%Y")
    # datetime_object_endDate = datetime.strptime(endDate, '%d/%m/%Y')

    # periodData = processedData.loc[(processedData['date'] <= datetime_object_endDate)
    #                                & (processedData['date'] >= datetime_object_startDate)]

    processedData = nameCategory(processedData)
    fig, (ax1) = plt.subplots(nrows=1)

    fig.set_size_inches(10, 8)
    if((typeHue=='duration' )|(typeHue == 'humidity')):
        sns.scatterplot(data=processedData, x="month", y="result",
                hue=typeHue, ax=ax1, color="blue", alpha=0.95)
    elif((typeHue=='Man')|(typeHue=='holiday')):
        sns.barplot(data=processedData, x="month", y="result",
                    hue=typeHue, ax=ax1, color="blue", alpha=0.95)
    elif(typeHue=='quarter'):
        sns.barplot(data=processedData, x="quarter", y="result",
                     ax=ax1, color="blue", alpha=0.95)

    ax1.legend(fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)

    result = toBase64(fig)

    plt.clf()
    plt.close()

    return result
