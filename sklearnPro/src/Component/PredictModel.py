from sysconfig import get_python_version
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
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
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBClassifier


def frameFromPeriod(year, month, Inspection, item, point):

    begin_date = str(year)+"-"+str(month)+"-01"
    end_date = str(year+1)+"-"+str(month)+"-01"
    df = pd.DataFrame({'Inspection Name': Inspection,
                       'point': point,
                       "item": item,
                       'forecast Date': pd.date_range(begin_date, end_date, freq='MS')})

    df["year"] = df["forecast Date"].dt.year
    df["month"] = df["forecast Date"].dt.month

    return df


def predictModel(data):
    max_depth_list = []

    # facilityModel = XGBClassifier()
    facilityModel = RandomForestRegressor(n_estimators=100,
                                          n_jobs=-1,
                                          random_state=0)

    categorical_feature_names = [
        "Inspection Name", "point", "item", "month", "year"]

    for var in categorical_feature_names:
        data[var] = data[var].astype("category")
#         test[var] = test[var].astype("category")

    X_train = data[categorical_feature_names]
    Y_train = data["result"]

    facilityModel.fit(X_train, Y_train)

    return facilityModel


def makePredictData(predictModel, year, month, Inspection, item, point):

    categorical_feature_names = [
        "Inspection Name", "point", "item", "month", "year"]
    predictDataFrame = frameFromPeriod(year, month, Inspection, item, point)
    ToPredictData = predictDataFrame[categorical_feature_names]

    predictDataFrame['prediction'] = predictModel.predict(ToPredictData)

    return predictDataFrame


def rmsle(predicted_values, actual_values):
    # ???????????? ?????? ????????? ????????????.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    # ???????????? ?????? ?????? 1??? ????????? ????????? ????????????.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    # ????????? ????????? ??????????????? ???????????? ????????? ????????? ?????????.
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)

    # ????????? ??????.
    mean_difference = difference.mean()

    # ?????? ????????? ?????????.
    score = np.sqrt(mean_difference)

    return score


def PredictByTwoTrainTest(train, test):
    max_depth_list = []

    facilityModel = RandomForestRegressor(n_estimators=100,
                                          n_jobs=-1,
                                          random_state=0)

    categorical_feature_names = [
        "Inspection Name", "point", "item", "month", "year"]

    for var in categorical_feature_names:
        train[var] = train[var].astype("category")
        test[var] = test[var].astype("category")

    X_train = train[categorical_feature_names]
    Y_train = train["result"]

    facilityModel.fit(X_train, Y_train)

    test = frameFromPeriod(1, 1, 1, 2022, 9, 2023, 9)
    ToPredictData = test[categorical_feature_names]

    predictions = facilityModel.predict(ToPredictData)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    # score = xgb.score(X_train, Y_train)

    get_python_version().run_line_magic('time',
                                        'score = cross_val_score(facilityModel, X_train, Y_train, cv=k_fold, scoring=rmsle_scorer)')

    # 0??? ??????????????? ?????? ?????????
    Per_score = cross_val_score(facilityModel, X_train, Y_train, cv=k_fold)
    return Per_score.mean()
