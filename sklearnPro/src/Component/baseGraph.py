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


def pointPlot(train):

    fig, (ax1) = plt.subplots(nrows=1)

    fig.set_size_inches(20, 10)

    sns.barplot(data=train, x="month", y="result",
                hue="Man", ax=ax1, color="blue", alpha=0.5)
    ax1.legend(fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    result = toBase64(fig)

    plt.clf()
    plt.close()

    return result
