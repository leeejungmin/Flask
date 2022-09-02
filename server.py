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
from sklearn.src.Component.compareGraph import compairPlot

from sklearn.src.Component.corMatt import corrMatt, findHighCorrList
from sklearn.src.Component.normalDistribution import normal_dis
from sklearn.src.Component.selectByMonthThree import combineHorizontalGraph
from sklearn.src.Component.baseGraph import pointPlot


def concatenate_point_item(train):
    return "{0}-{1}".format(train["point"], train["item_name"])


def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)


mpl.use('Agg')
app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

mpl.rcParams['axes.unicode_minus'] = False

train = pd.read_csv("data/data/team5train2.csv", parse_dates=["date"])
train.shape
test = pd.read_csv("data/data/team5test2.csv", parse_dates=["date"])

train.head()

# train = train.loc[(train["Inspection Name"] == "Hydraulic System Check") & (
#     train["point"] == "coupling")]
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month


train["year_month"] = train["date"].apply(concatenate_year_month)

categorical_feature_names = ["Inspection Name", "holiday",
                             "Man", "duration", "humidity", "item", "date", "result"]
title_mapping = {"Hydraulic System Check": 1, "Camshaft": 2, "Cylinder": 3, "Engine Check": 4,
                 "Table Roller": 5, "Reducer Check": 6, "FuelRail": 7, "InterCooler": 8, "Geared Motor Check": 9}
point_mapping = {'coupling': 1, 'oil': 2, 'tank': 3, ' in the filter': 4, 'fuelRail': 5, 'bearing': 6,
                 'same position(vertical)': 7, 'bearing cover': 8, 'gear': 9, 'coolant': 10, 'cooling fan': 11, 'roller': 12, 'in the filter': 13, 'pump': 14, 'o-ring': 15}

train['Inspection Name'] = train['Inspection Name'].map(title_mapping)
test['Inspection Name'] = test['Inspection Name'].map(title_mapping)
train['point'] = train['point'].map(point_mapping)
test['point'] = test['point'].map(point_mapping)


dr = pd.date_range(start='2018-01-01', end='2022-12-31')
df = pd.DataFrame()
df['Date'] = dr
cal = calendar()
holidays = cal.holidays(start=dr.min(), end=dr.max())

train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek
train['holiday'] = train['date'].dt.date.astype('datetime64').isin(holidays)

test["year"] = test["date"].dt.year
test["month"] = test["date"].dt.month
test["day"] = test["date"].dt.day
test["dayofweek"] = test["date"].dt.dayofweek
test['holiday'] = test['date'].dt.date.astype('datetime64').isin(holidays)

train.loc[train["holiday"] == True, "holiday"] = 0
train.loc[train["holiday"] == False, "holiday"] = 1

train.loc[train["humidity"] == 0, "humidity"] = train["humidity"].mean()

test.loc[test["holiday"] == True, "holiday"] = 0
test.loc[test["holiday"] == False, "holiday"] = 1


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


def combineGraph(data, c_year, c_month, Inspection, item, point):

    bar_width = 0.3
    alpha = 0.39

    c_year = c_year
    p_year = c_year-1
    c_month = c_month
    n_month = c_month + 1

    prev_port = str(p_year)+"-"+str(c_month)
    cur_port = str(c_year)+"-"+str(c_month)
    next_port = str(c_year)+"-"+str(c_month+1)
    dst_ports = str(p_year)+"-"+str(c_month), str(c_year) + \
        "-"+str(c_month), str(c_year)+"-"+str(c_month+1)

    xlabel = [0, 1, 2, 3, 4]

    standard = 59
    standard_point = [standard]*5
    plt.plot(xlabel, standard_point, color='firebrick', alpha=0.5)

    label = [prev_port, cur_port, next_port]

    prev_port_count = selected_num(
        data, c_year-1, c_month, Inspection, item, point)
    cur_port_count = selected_num(
        data, c_year, c_month, Inspection, item, point)
    next_port_count = selected_num(
        data, c_year, c_month+1, Inspection, item, point)

    prev_c = 'b'
    cur_c = 'b'
    next_c = 'b'

    print('prev_port_count', prev_port_count)
    if prev_port_count > standard:
        prev_c = 'firebrick'
    if cur_port_count > standard:
        cur_c = 'firebrick'
    if next_port_count > standard:
        next_c = 'firebrick'

    p1 = plt.bar(1, prev_port_count,
                 bar_width,
                 color=prev_c,
                 alpha=alpha,
                 label=prev_port)

    p2 = plt.bar(2,
                 cur_port_count,
                 bar_width,
                 color=cur_c,
                 alpha=alpha,
                 label=cur_port)

    p3 = plt.bar(3,
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

    plt.title('hydraulic System',  fontsize=10)
    plt.ylabel(' ', fontsize=8)
    plt.xlabel('Month', fontsize=7)
    plt.xticks(x, label, fontsize=8)

    plt.legend((p1[0], p2[0], p3[0]), (prev_port,
               cur_port, next_port), fontsize=10)

    return toBase64(plt)


def linegraph(test):
    alpha = 0.3
    color = 'blue'
    trainWCondition = test.loc[(test["year"] == 2019) & (
        test["Inspection Name"] == 1) & (test["item"] == 2) & (test["point"] == 1)]

    fig, (ax1) = plt.subplots(nrows=1)
    fig.set_size_inches(5, 5)
    sns.lineplot(data=trainWCondition, x="month", y="result",
                 color=color, ax=ax1, alpha=alpha)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    return toBase64(fig)


def showThreePlot(data, c_year, c_month, Inspection, item, point):
    c_year = c_year
    p_year = c_year-1
    c_month = c_month
    n_month = c_month + 1

    prev_port = [str(p_year)+"-"+str(c_month)]
    cur_port = [str(c_year)+"-"+str(c_month)]
    next_port = [str(c_year)+"-"+str(c_month+1)]
    dst_ports = [str(p_year)+"-"+str(c_month), str(c_year) +
                 "-"+str(c_month), str(c_year)+"-"+str(c_month+1)]

    prev_port_count = [selected_num(
        data, c_year-1, c_month, Inspection, item, point)]
    cur_port_count = [selected_num(
        data, c_year, c_month, Inspection, item, point)]
    next_port_count = [selected_num(
        data, c_year, c_month+1, Inspection, item, point)]

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.ylim(0, 150)
    plt.figure(figsize=(5, 2))
    bar_width = 0.35
    x = np.arange(1, len(dst_ports)+2)

    ax1.bar(prev_port, prev_port_count,
            color="blue", alpha=0.5, width=0.1)
    ax1.set(ylabel='prediction')

    ax1.set_ylim(0, 150)
    plt.legend(fontsize=15)

    ax2.bar(cur_port, cur_port_count,
            color="blue", alpha=0.5, width=0.1)
    ax2.set()
    ax2.set_ylim(0, 150)
    ax2.legend(fontsize=5)

    ax3.bar(next_port, next_port_count,
            color="red", alpha=0.5, width=0.1)
    ax3.set()
    ax3.set_ylim(0, 150)
    ax3.legend(fontsize=5)

    plt.grid(b=None)
    # figure.ax1.get_yaxis().set_visible(False)
    return toBase64(figure)


def pointPlotByYearMonth(train, Year, Month, Inspection, Item):

    train = train.loc[(train['Inspection Name'] == Inspection)
                      & (train['item'] == Item)]

    year = str(Year)  # year = "2020"
    if(Month < 10):
        Month = "0"+str(Month)

    s_month = str(Month)  # s_month = "08"

    e_month = "12"  # e_month = "12"
    start_date = year+"-"+s_month+"-01"
    end_date = year+"-"+e_month+"-31"

    train = train[train["date"].isin(
        pd.date_range(start_date, end_date))]

    fig, (ax1) = plt.subplots(nrows=1)
    fig.set_size_inches(20, 10)
    sns.barplot(data=train, x="month", y="result",
                hue="Man", ax=ax1, color="blue", alpha=0.5)
    plt.legend(fontsize=14)
    # sns.barplot(data=train, x="month", y="result", hue="item", ax=ax2)

    # sns.pointplot(data=train, x="month", y="result", hue="duration", ax=ax3)

    # sns.pointplot(data=train, x="month", y="result", hue="point", ax=ax4)

    return toBase64(fig)


def pointPlotByYearMonth2(train, Year, Month, Inspection, Item):

    train = train.loc[(train['month'] == Month) &
                      (train['year'] == Year) & (train['Inspection Name'] == Inspection) & (train['item'] == Item)]

    year = str(Year)  # year = "2020"
    if(Month < 10):
        Month = "0"+str(Month)

    s_month = str(Month)  # s_month = "08"

    e_month = "12"  # e_month = "12"
    start_date = year+"-"+s_month+"-01"
    end_date = year+"-"+e_month+"-31"

    train = train[train["date"].isin(
        pd.date_range(start_date, end_date))]

    fig, (ax1) = plt.subplots(nrows=1)
    fig.set_size_inches(20, 10)
    sns.barplot(data=train, x="month", y="result",
                hue="Man", ax=ax1, color="blue", alpha=0.5)

    return toBase64(fig)


def linePlot(train):

    fig, (ax1) = plt.subplots(nrows=1)

    fig.set_size_inches(15, 5)
    # sns.set_theme(style="darkgrid")
    sns.barplot(data=train, x="month", y="result",
                hue="Man", ax=ax1, color="blue", alpha=0.5)
    ax1.legend(fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # sns.barplot(data=train, x="month", y="result", hue="item", ax=ax2)

    # sns.pointplot(data=train, x="month", y="result", hue="duration", ax=ax3)

    # sns.pointplot(data=train, x="month", y="result", hue="point", ax=ax4)

    return toBase64(fig)


def selected_season(train, Year, Season, Inspection, point, item):
    predicted_num = train.loc[(train["year"] == Year) &
                              (train["Inspection Name"] == Inspection) &
                              (train["season"] == Season) &
                              (train["point"] == point) &
                              (train["item"] == item)]

    predicted_num['result'] = predicted_num['result'].astype(int).mean()

    return (predicted_num['result'])


def showSeason(data, Year, Inspection, item, point):

    train = data.loc[(data["year"] == Year) &
                     (data["Inspection Name"] == Inspection)]

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.set_size_inches(18, 10)

    sns.barplot(data=train, x="season", y="result", hue="Man", ax=ax1)

    sns.barplot(data=train, x="season", y="result", hue="item", ax=ax2)

    return fig


def SeasonGraph(data, Year, Inspection, item, point):

    bar_width = 0.35
    alpha = 0.4

    p0_count = selected_season(
        data, Year, 0, Inspection, point, item).astype(int).mean()
    p1_count = selected_season(
        data, Year, 1, Inspection, point, item).astype(int).mean()
    p2_count = selected_season(
        data, Year, 2, Inspection, point, item).astype(int).mean()
    p3_count = selected_season(
        data, Year, 3, Inspection, point, item).astype(int).mean()

    c_year = Year
    p_year = Year-1

    prev_port = "spring"
    cur_port = "summer"
    next_port = "autumn"
    dst_ports = "winter"

    xlabel = ["", prev_port, cur_port, next_port, dst_ports]

    standard = 55
    standard_point = [standard]*5

    plt.plot(xlabel, standard_point)

    prev_c0 = 'b'
    prev_c1 = 'b'
    prev_c2 = 'b'
    prev_c3 = 'b'

    if p0_count > standard:
        prev_c0 = 'r'
    if p1_count > standard:
        prev_c1 = 'r'
    if p2_count > standard:
        prev_c2 = 'r'
    if p3_count > standard:
        prev_c3 = 'r'

    p0 = plt.bar(1,
                 p0_count,
                 bar_width,
                 color=prev_c0,
                 alpha=alpha,
                 label=prev_port)

    p1 = plt.bar(2,
                 p1_count,
                 bar_width,
                 color=prev_c1,
                 alpha=alpha,
                 label=cur_port)

    p2 = plt.bar(3,
                 p2_count,
                 bar_width,
                 color=prev_c2,
                 alpha=alpha,
                 label=next_port)

    p3 = plt.bar(4,
                 p3_count,
                 bar_width,
                 color=prev_c3,
                 alpha=alpha,
                 label=dst_ports)

    x = np.arange(0, 3+2)
    print(x)
    plt.title(Year, fontsize=20)
    plt.ylabel('Predict', fontsize=18)
    plt.xlabel('', fontsize=18)
    plt.xticks(x, xlabel, fontsize=15)

    plt.legend((p0[0], p1[0], p2[0], p3[0]), (prev_port,
               cur_port, next_port, dst_ports), fontsize=15)

    return toBase64(plt)


@ app.route("/SeasonGraph",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def SeasonGraph():
    if request.method == 'POST':

        data = request.get_json()["data"]
        Year = int(data['selectY'])
        Month = int(data['selectM'])
        Inspection = int(data['selectN'])
        Point = int(data['selectP'])
        Item = int(data['selectI'])

        SeasonGraphImage = SeasonGraph(train, Year, Inspection, Item, Point)

    return {"base64": SeasonGraphImage}


@ app.route("/normalDis",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def normalDis():
    if request.method == 'POST':
        data = request.get_json()["data"]
        Year = int(data['selectY'])
        Month = int(data['selectM'])
        Inspection = int(data['selectN'])
        Point = int(data['selectP'])
        Item = int(data['selectI'])

        normalDisImage = normal_dis(train, Inspection, Item, Point)

    return {"base64": normalDisImage}


@ app.route("/corrMattImage",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def corrMattImage():
    if request.method == 'POST':
        corrMattimage = corrMatt(train)

    return {"base64": corrMattimage}


@ app.route("/selectByYear",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def selectByYear():
    if request.method == 'POST':

        data = request.get_json()["data"]
        Year = int(data['selectY'])
        Month = int(data['selectM'])
        Inspection = int(data['selectN'])
        Point = int(data['selectP'])
        Item = int(data['selectI'])

        pointImage = pointPlotByYearMonth(
            train, Year, Month, Inspection, Item)

    # return jsonify(data), 200
    return {"base64": pointImage}


@ app.route("/baseGraph",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def baseGraph():
    if request.method == 'GET':
        time.sleep(1.5)
        barImage = pointPlot(train)

    return {"base64": barImage}


@ app.route("/selectByMonthThree",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def selectByMonthThree():
    if request.method == 'GET':

        # data = request.get_json()

        # print('selectByMonthThree.....', data)
        # Year = int(data['selectY'])
        # Month = int(data['selectM'])
        # Inspection = int(data['selectN'])
        # Point = int(data['selectP'])
        # Item = int(data['selectI'])

        Year = 2021
        Month = 8
        Inspection = 1
        Item = 2
        Point = 1

        combineHorizontalGraphImage = combineHorizontalGraph(
            test, Year, Month, Inspection, Item, Point)
        # time.sleep(2.5)
    return {"base64": combineHorizontalGraphImage}


@ app.route("/compairPlotGraph",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def compairPlotGraph():
    if request.method == 'POST':

        # data = request.get_json()

        # print('selectByMonthThree.....', data)
        # Year = int(data['selectY'])
        # Month = int(data['selectM'])
        # Inspection = int(data['selectN'])
        # Point = int(data['selectP'])
        # Item = int(data['selectI'])

        Year = 2021
        Month = 8
        Inspection = 1
        xVariable = "humidity"
        yVariable = "duration"

        compairPlotImage = compairPlot(
            test, xVariable, yVariable, Inspection)

    return {"base64": compairPlotImage}


@ app.route("/findHighCorr",  methods=['GET', 'POST'])
@ cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def findHighCorr():
    if request.method == 'POST':

        # data = request.get_json()

        # print('selectByMonthThree.....', data)
        # Year = int(data['selectY'])
        # Month = int(data['selectM'])
        # Inspection = int(data['selectN'])
        # Point = int(data['selectP'])
        # Item = int(data['selectI'])

        Year = 2021
        Month = 8
        Inspection = 1
        xVariable = "humidity"
        yVariable = "duration"

        findHighCorrImage = findHighCorrList(test)

    return {"base64": findHighCorrImage}


@ app.route("/jung")
def jung():
    return {"good": ["lee", "jung", "min"]}


@ app.route("/min")
def min():
    return {"result": train.head(10)}


if __name__ == "__main__":
    app.run(threaded=False, debug=True)
    #  app.run(threaded=False, debug=True,  processes=5)
