from flask import Flask, send_file
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import base64
import json
import io
from flask_cors import CORS, cross_origin


def concatenate_point_item(train):
    return "{0}-{1}".format(train["point"], train["item_name"])


def concatenate_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)


app = Flask(__name__)

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

mpl.rcParams['axes.unicode_minus'] = False

train = pd.read_csv("data/data/team5train2.csv", parse_dates=["date"])
train.shape

train.head()

train = train.loc[(train["Inspection Name"] == "Hydraulic System Check") & (
    train["point"] == "coupling")]
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month


train["year_month"] = train["date"].apply(concatenate_year_month)


# print(train.shape)
# train[["date", "year_month"]].head()


# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_size_inches(12, 10)
# sns.boxplot(data=train, y="result", orient="v", ax=axes[0][0])
# sns.boxplot(data=train, y="result", x="Man", orient="v", ax=axes[0][1])
# sns.boxplot(data=train, y="result", x="humidity", orient="v", ax=axes[1][0])
# sns.boxplot(data=train, y="result", x="holiday", orient="v", ax=axes[1][1])

# axes[0][0].set(ylabel='result', title="Q")
# axes[0][1].set(xlabel='Man', ylabel='result', title="Man")
# axes[1][0].set(xlabel='humidity', ylabel='result', title="humidity")
# axes[1][1].set(xlabel='holiday', ylabel='result', title="holiday")

# pic_IObytes = io.BytesIO()
# in_fig = fig
# in_fig.savefig(pic_IObytes,  format='png')
# pic_IObytes.seek(0)
# pic_hash = base64.b64encode(pic_IObytes.read())


def pointPlot(train):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

    fig.set_size_inches(35, 40)

    sns.barplot(data=train, x="month", y="result", hue="Man", ax=ax1)

    sns.barplot(data=train, x="month", y="result", hue="item", ax=ax2)

    sns.pointplot(data=train, x="month", y="result", hue="duration", ax=ax3)

    sns.pointplot(data=train, x="month", y="result", hue="point", ax=ax4)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


@app.route("/image",  methods=['GET'])
@cross_origin(origin='*', headers=['access-control-allow-origin', 'Content- Type', 'Authorization'])
def image():
    pointimage = pointPlot(train)

    # return {"base64": figure}
    return {"base64": pointimage}
    # return send_file(filename, mimetype='png')


@app.route("/jung")
def jung():
    return {"good": ["lee", "jung", "min"]}


@app.route("/min")
def min():
    return {"result": train.head(10)}


if __name__ == "__main__":
    app.run(debug=True)
