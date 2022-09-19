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


def selected_num(train, Year, Month, Inspection, item, point):
    predicted_num = train.loc[(train["year"] == Year) &
                              (train["Inspection Name"] == 1) &
                              (train["month"] == Month) &
                              (train["item"] == 2) &
                              (train["point"] == 1)]
    return (int(predicted_num['result']))


def combineHorizontalGraph(data, c_year, c_month, Inspection, item, point):

    plt.show()

    plt.figure(figsize=(10, 5))
    bar_width = 0.39
    alpha = 0.7

    c_year = c_year
    p_year = c_year-1
    c_month = c_month
    n_month = c_month + 1

    prev_port = str(p_year)+"-"+str(c_month)
    cur_port = str(c_year)+"-"+str(c_month)
    next_port = str(c_year)+"-"+str(c_month+1)
    dst_ports = str(p_year)+"-"+str(c_month), str(c_year) + \
        "-"+str(c_month), str(c_year)+"-"+str(c_month+1)

    prev_port_label = "In Safe"
    cur_port_label = "In Safe"
    next_port_label = "In Safe"

    xlabel = [0, 1, 2, 3, 4]

    standard = 59
    standard_point = [standard]*5
#     plt.ayline(xlabel, standard_point, color='red', alpha=0.5)
    plt.axvline(x=standard, linewidth=3, color='orange', linestyle='--')

    label = [prev_port, cur_port, next_port]

    prev_port_count = selected_num(
        data, c_year-1, c_month, Inspection, item, point)
    cur_port_count = selected_num(
        data, c_year, c_month, Inspection, item, point)
    next_port_count = selected_num(
        data, c_year, c_month+1, Inspection, item, point)
    # red_color = 'firebrick'
    red_color = 'orange'

    prev_c = 'b'
    cur_c = 'b'
    next_c = 'b'

    print('prev_port_count', prev_port_count)
    if prev_port_count > standard:
        prev_c = red_color
        prev_port_label = "In Danger"

    if cur_port_count > standard:
        cur_c = red_color
        cur_port_label = "In Danger"

    if next_port_count > standard:
        next_c = red_color
        next_port_label = "In Danger"

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

    x = np.arange(1, len(dst_ports)+1)

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
