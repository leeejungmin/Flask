import re
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


mpl.use('Agg')


def toBase64(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    return pic_hash.decode('utf-8')


def corrMatt(train, Inspection):
    processed_data = train.loc[
        (train["Inspection Name"] == Inspection)

    ]
    corrMatt = processed_data[["holiday", "Man",
                               "duration", "humidity", "item", "result"]]
    corrMatt = corrMatt.corr()
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False

    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    sns.heatmap(corrMatt, mask=mask, vmax=.8,
                square=True, annot=True, cmap='Blues', annot_kws={
                    'fontsize': 8,
                    'fontweight': 'bold',
                    'fontfamily': 'arial'
                })
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    # plt.xlabel(fontsize=5)
    # plt.ylabel(fontsize=5)
    plt.xticks(rotation=360, fontsize=8)
    plt.yticks(rotation=360, fontsize=8)

    result = toBase64(fig)

    plt.clf()
    plt.close()

    return result


def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj


def findHighCorrList(train, Inspection):
    processed_data = train.loc[
        (train["Inspection Name"] == Inspection)

    ]
    corrMatt = processed_data[["holiday", "Man",
                               "duration", "humidity", "item", "result"]]
    corrMattAfterCorr = corrMatt.corr()
    filteredCorMatt = corrMattAfterCorr[(corrMattAfterCorr != 1.0) & (corrMattAfterCorr >= .2) | (
        corrMattAfterCorr <= -.2)]
    corrStack = filteredCorMatt.unstack()
    corrByQuickSort = corrStack.sort_values(kind="quicksort")
    removeddupli_corr = corrByQuickSort.dropna().drop_duplicates()
    find5list_corr = removeddupli_corr.keys()[:8]
    set_corr = {(i, j) for i, j in find5list_corr}
    tojson_corr = json.dumps(set_corr, default=serialize_sets)
    result = re.sub("'", "", tojson_corr)

    return result
