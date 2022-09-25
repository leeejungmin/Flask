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
mpl.use('Qt5Agg')


def compairPlot(train, xVariable, yVariable, Inspection):
    processed_data = train.loc[
        (train["Inspection Name"] == Inspection)

    ]
    sns.scatterplot(x=xVariable, y=yVariable, data=processed_data,
                    s=200)
    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    sns.set(rc={'figure.figsize': (5, 6)})

    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.set_context("paper")

    plt.xlabel(xVariable, fontsize=14)
    plt.ylabel(yVariable, fontsize=14, rotation=90)
    plt.show()

    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result
