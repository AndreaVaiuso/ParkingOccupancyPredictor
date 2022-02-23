import numpy
import matplotlib.pyplot as plt
from utilities import bcolors, secToTime
import math
import csvtools as ct
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys
import os
import calendar
import json

SECONDS_IN_WEEK = 604800
SECONDS_IN_DAY = 86400
SAMPLING_TIME = 3600
LOOK_BACK = 3
WEEKS_MEAN_NUMBER = 4

DEBUG = False

def debug(string):
    if DEBUG: 
        print(bcolors.WARNING+"---DEBUG----"+bcolors.ENDC)
        print(bcolors.WARNING+str(string)+bcolors.ENDC)
        print(bcolors.WARNING+"------------"+bcolors.ENDC)

def to_lstm_input_shape(data):
    return numpy.reshape(data, (data.shape[0], 1, data.shape[1]))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return numpy.array(dataX), numpy.array(dataY)

def printRow(data):
    print(data['DATE'],data['WEEKDAY'],data['HOUR_OF_THE_DAY'],data['TOTAL_OCCUPIED_RATIO'])

def getTrends(data_frame,weeks_number,week_day,look_back=0,samplingTime=3600,label_column="TOTAL_OCCUPIED_RATIO",verbose=False):
    i = -1
    week_day = str(week_day)
    last_day = data_frame[i]["WEEKDAY"]
    if week_day==last_day:
        i -= int(SECONDS_IN_WEEK / samplingTime) - int(SECONDS_IN_DAY/samplingTime)
    else:
        while True:
            i -= 1
            if data_frame[i]["WEEKDAY"] == week_day:
                break
    trend = []
    stop = False
    for week in range(weeks_number):
        fr = i + look_back
        to = i - int(SECONDS_IN_DAY/samplingTime) - look_back - 1
        temp_tr = []
        for j in range(fr,to,-1):
            try:
                temp_tr.append(float(data_frame[j][label_column]))
            except IndexError:
                if verbose: print("Reached maximum week limit. Data truncated at value: "+str(week))
                stop = True
                break
        if stop: break
        trend.append(temp_tr)
        trend[-1].reverse()
        i -= int(SECONDS_IN_DAY/samplingTime)
        i -= int(SECONDS_IN_WEEK / samplingTime) - int(SECONDS_IN_DAY/samplingTime)
    return trend

def mergeData(trend,look_back=0):
    data = []
    for i in range(len(trend)):
        for j in range(look_back,len(trend[i])-look_back):
            data.append(trend[i][j])
    return data

def convertToColumn(data):
	column = []
	for i in range(len(data)):
		column.append([data[i]])
	return numpy.array(column)

def splitData(data, splitVal = 0.7):
	n = len(data)
	train_ds = data[0:int(n * splitVal)]
	val_ds = data[int(n * splitVal):n]
	return train_ds, val_ds

def select(data,length):
    dat = []
    for i in range(length):
        dat.append(data)
    return dat

def generateWeights(size,firstElementGain=1):
    weights = []
    for x in range(size):
        val = -math.atan(x)+(math.pi/2)
        if x == 0: val *= firstElementGain
        weights.append(val)
    nfac = 1/sum(weights)
    for i in range(len(weights)):
        weights[i] = weights[i] * nfac
    return weights

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def weightedMean(data_array,weights):
    weighted_data = [0] * len(data_array[0])
    for i in range(len(data_array[0])):
        for j in range(len(weights)):
            weighted_data[i]+= data_array[j][i] * weights[j]
    return weighted_data
    
def padding(array,size=1):
    ar = []
    for i in range(size):
        ar.append([numpy.nan])
    for i in range(len(array)):
        ar.append([array[i][0]])
    return ar

def integers_list(a, b):
    return [i for i in range(a, b+1)]

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def create_lstm(look_back):
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model, early_stopping

def do_train(history_trend,week_day,pklot_ID,look_back):
    train = mergeData(history_trend,look_back)
    train = convertToColumn(train)
    trainX, trainY = create_dataset(train,look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model, early_stopping = create_lstm(look_back)
    model_f_name = "NETWORK/"+str(pklot_ID)+"/"+calendar.day_abbr[week_day]+".keras"
    model.fit(trainX, trainY, validation_split=0.3, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
    model.save(model_f_name)
    return model

def load_model(week_day,pklot_ID,look_back):
    model_f_name = "NETWORK/"+str(pklot_ID)+"/"+calendar.day_abbr[week_day]+".keras"
    model, _ = create_lstm(look_back)
    try:
        model.load_weights(model_f_name)
    except:
        print("Error while loading weights for the model: " + model_f_name)
        return None
    return model

def generate_models_array(df,weeks_train_number,pklot_ID,look_back):
    models = []
    trends = []
    for week_day in range(7):
        history_trend = getTrends(df,weeks_train_number,week_day,look_back)
        model = load_model(week_day,pklot_ID,look_back)
        if model is None:
            model = do_train(history_trend,week_day,pklot_ID,look_back)
        models.append(model)
        trends.append(history_trend)
    return models, trends

def generate_model(df,weeks_train_number,pklot_ID,week_day,look_back,force_train=False):
    history_trend = getTrends(df,weeks_train_number,week_day,look_back,samplingTime=SAMPLING_TIME)
    model = None
    if force_train:
        model = do_train(history_trend,week_day,pklot_ID,look_back)
        return model, history_trend
    model = load_model(week_day,pklot_ID,look_back)
    if model is None:
        model = do_train(history_trend,week_day,pklot_ID,look_back)
    return model, history_trend


def predict(model,trend_array,weeks_mean_number=WEEKS_MEAN_NUMBER,look_back=LOOK_BACK,weeks_shift=0,ground_truth=None):
    weights = generateWeights(weeks_mean_number)
    trend = trend_array
    mse = numpy.nan
    n_prediction = []
    for i in range(weeks_shift+1):
        n = len(trend)
        w_set = weightedMean(trend[n-weeks_mean_number:n],weights)
        pad_head = w_set[0:look_back]
        pad_teal = w_set[-1]
        w_set = convertToColumn(w_set)
        testX, _ = create_dataset(w_set,look_back)
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        n_prediction = model.predict(testX)
        n_prediction = padding(n_prediction,look_back)
        for i in range(len(n_prediction)):
            val = n_prediction[i][0]
            if not numpy.isnan(val):
                pad_head.append(val)
        pad_head.append(pad_teal)
        trend.append(pad_head)
    prediction = n_prediction[look_back:len(n_prediction)-look_back+1]
    if ground_truth is not None:
        ground_truth = convertToColumn(ground_truth)
        mse = math.sqrt(mean_squared_error(ground_truth[0:len(ground_truth)][0], n_prediction[0:len(n_prediction)][0]))
    return prediction, mse

def prepare(week_day,pklot_ID):
    df = ct.csv_open("DATASET/PARKING_LOTS/"+str(pklot_ID)+"/history.csv",sep=";").getDataFrame()
    weeks_train_number = 21
    model, trend = generate_model(df,weeks_train_number,pklot_ID,week_day,LOOK_BACK)
    return model, trend

def getCurrent(pklot_ID):
    ds = ct.csv_open("DATASET/PARKING_LOTS/"+str(pklot_ID)+"/current_day.csv").getDataFrame()
    date = str(ds[0]["DATE"])
    resp = {}
    resp["SAMPLING_TIME"] = 3600
    resp["DATE"] = date
    resp["WEEKDAY"] = str(ds[0]["WEEKDAY"])
    for d in ds:
        ext_hodd = d["HOUR_OF_THE_DAY"]
        x = ext_hodd.strip().split(":")
        hodd = x[0] + ":" + x[1] 
        resp[hodd] = d["TOTAL_OCCUPIED_RATIO"]
    jsonobj = json.dumps(resp)
    return jsonobj

def makePrediction(index,week_shift,pklot_ID):
    if index not in range(7):
        return json.dumps({"status": "failed"})
    print("Request: ",index," Week_shift: ",week_shift," POI: ",pklot_ID)
    model,trend = prepare(index,pklot_ID) 
    prediction, _ = predict(model,trend,weeks_shift=week_shift)
    pr = []
    for pred in prediction:
        pr.append(pred[0])
    out = {}
    t = 0
    for pred in pr:
        out[secToTime(t,clockFormat=True)] = str(pred)
        t += SAMPLING_TIME
    jsonobj = json.dumps(out)
    return jsonobj

def main(week_day,pklot_ID):
    df = ct.csv_open("DATASET/PARKING_LOTS/"+str(pklot_ID)+"/history.csv",sep=";").getDataFrame()
    weeks_train_number = 21
    look_back = 3
    model,trend = generate_model(df,weeks_train_number,pklot_ID,week_day=week_day,look_back=look_back)
    plt.xlim([0,23])
    plt.ylim([0,1])
    for i in range(4):
        plt.plot(trend[i][look_back:len(trend[i])-look_back])
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    prediction, mse = predict(model,trend,weeks_mean_number=4,look_back=look_back)
    pr = []
    for pred in prediction:
        pr.append(pred[0])
    out = {}
    t = 0
    for pred in pr:
        out[secToTime(t,clockFormat=True)] = str(pred)
        t += SAMPLING_TIME
    json_object = json.dumps(out, indent = 4)
    plt.xlim([0,23])
    plt.ylim([0,1])
    plt.plot(prediction,label="prediction")
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close() 

if __name__ == "__main__":
    for i in range(7):
        main(i,24)