import tkinter
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
mpl.use('Qt5Agg')


def toBase64(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


def compairPlot(train, xVariable, yVariable, Inspection):
    processed_data = train.loc[
        (train["Inspection Name"] == Inspection)

    ]
    sns.scatterplot(x=xVariable, y=yVariable, data=processed_data
                    )
    sns.set_theme(font='Arial',
                  rc={'axes.unicode_minus': False},
                  style='white',
                  )
    sns.set_style(style='white')
    sns.set(rc={'figure.figsize': (7, 9)})

    plt.xticks(rotation=30, fontsize=10)
    plt.yticks(rotation=30, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.set_context("paper", rc={"font.size": 14,
                    "axes.titlesize": 14, "axes.labelsize": 14})

    plt.show()

    result = toBase64(plt)

    plt.clf()
    plt.close()

    return result
