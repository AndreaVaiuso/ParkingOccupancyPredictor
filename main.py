import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
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

SECONDS_IN_WEEK = 604800
SECONDS_IN_DAY = 86400

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return numpy.array(dataX), numpy.array(dataY)

def getTrends(data_frame,weeks_number,look_back,samplingTime=3600,label_column="TOTAL_OCCUPIED_RATIO"):
    i = -1
    last_day = data_frame[i]["WEEKDAY"]
    last_day_trend = [float(data_frame[i][label_column])]
    while i>-len(data_frame):
        i-=1
        if data_frame[i]["WEEKDAY"]==last_day:
            last_day_trend.append(float(data_frame[i][label_column]))
        else:
            i += 1
            break
    trend = []
    stop = False
    for week in range(weeks_number):
        trend.append([])
        i -= int(SECONDS_IN_WEEK / samplingTime) - int(SECONDS_IN_DAY/samplingTime)
        fr = (i-1) + look_back
        to = (i-int(SECONDS_IN_DAY/samplingTime)-1) - look_back
        for j in range(fr,to,-1):
            try:
                trend[week].append(float(data_frame[j][label_column]))
            except IndexError:
                print("Reached maximum week limit. Data truncated at value: "+str(j))
                stop = True
                break
        trend[week].reverse()
        i -= int(SECONDS_IN_DAY/samplingTime)
        if stop: break
    last_day_trend.reverse()
    return trend, last_day_trend, last_day

def mergeData(trend,look_back):
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

def compute(file,to=11,test_num=7):
    # L'ultima riga del file rappresenta il giorno e l'ora corrente (ultima misurazione effettuata)
    df = ct.csv_open("DATASET/"+file,sep=";")
    weeks_train_number = 21
    look_back = 3
    # history_trend contiene i dati settimanali ordinati dal più recente al meno recente, dalle 00:00 alle 23:00
    # current_day_trend contiene i dati della settimana corrente da 00:00 all'ora corrente (ultima nel file csv)
    # current_weekday rappresenta il numero del giorno della settimana corrente (ultima nel file csv)
    history_trend, current_day_trend, current_weekday = getTrends(df,weeks_train_number,look_back)
    
    current_day_len = len(current_day_trend)

    # unisco in un solo vettore tutti i dati inerenti le settimane precedenti, rimuovendo il padding
    dataset = mergeData(history_trend,look_back)
    # preparo training set e validation set
    train, valid = splitData(dataset,0.7)
    # converto nel formato compatibile di TensorFlow
    train = convertToColumn(train)
    valid = convertToColumn(valid)
    dataset = convertToColumn(dataset)
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    # reshape input to be [samples, time steps, features], compatible with LSTM input format
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    validX = numpy.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    # create and fit the LSTM network

    model = Sequential()
    model.add(LSTM(32, input_shape=(1, look_back)))
    model.add(Dense(1))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3,mode='min')
    model.compile(loss='mean_squared_error', optimizer='adam')
    net_file_name = "NETWORK/"+file+".keras"
    if not os.path.isfile(net_file_name):
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
        model.save(net_file_name)
    else:
        model.load_weights(net_file_name)
    fr = 1
    msemean = [0] * (to - fr)
    for weeks_mean_number in range(fr,to):
        weights = generateWeights(weeks_mean_number)
        num = 0
        for j in range(-1,-num-(test_num+1),-1):
            n = len(history_trend)
            w_set = weightedMean(history_trend[n+j-weeks_mean_number:n+j],weights)
            pred_day = convertToColumn(history_trend[n+j-weeks_mean_number-1][look_back:len(history_trend[n+j-weeks_mean_number-1])-look_back])
            w_set = convertToColumn(w_set)
            testX, _ = create_dataset(w_set,look_back)
            testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
            n_prediction = model.predict(testX)
            n_prediction = padding(n_prediction,look_back)
            n_prediction = n_prediction[look_back:len(n_prediction)-look_back+1]
            mse = math.sqrt(mean_squared_error(pred_day[0:len(pred_day)][0], n_prediction[0:len(n_prediction)][0]))
            msemean[(weeks_mean_number - fr)] += mse
            num += 1
        print(weeks_mean_number," -> ", num)
        msemean[(weeks_mean_number - fr)] /= num
    print("Done")
    for i in range(len(msemean)):
        err = str(msemean[i])
        print("MSE using " + str(i+1) + " weeks average:\t["+err+"]")
    return msemean
    """""
    # calculate weighted mean data
    weeks_mean_number = 4
    weights = generateWeights(weeks_mean_number)
    w_set = weightedMean(history_trend,weights)
    current_day_trend_col = convertToColumn(current_day_trend)
    w_set = convertToColumn(w_set)
    testX, _ = create_dataset(w_set,look_back)
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # make prediction
    n_prediction = model.predict(testX)
    n_prediction = padding(n_prediction,look_back)
    n_prediction = n_prediction[look_back:len(n_prediction)-look_back+1]
    # plot data
    showPlot = False
    plt.xlim([0,23])
    plt.ylim([0,1])
    if showPlot:
        for i in range(weeks_mean_number):
            plt.plot(history_trend[i][look_back:len(history_trend[i])-look_back],label="current week "+str(-i))
        plt.plot(w_set[look_back:len(w_set)-look_back],label="mean",linewidth=3)
        plt.legend()
        plt.show()

    plt.plot(n_prediction[look_back:len(n_prediction)-look_back+1],label="prediction")
    plt.plot(w_set[look_back:len(w_set)-look_back],label="Weighted mean")
    plt.plot(current_day_trend_col,label="ground truth")
    
    plt.legend()
    plt.show()
    """""


if __name__ == "__main__":
    files = os.listdir("DATASET")
    errors = []
    files = sorted(files)
    to = 8
    for file in files:
        if file.endswith(".csv"):
            print("Using: "+file)
            errors.append(compute(file,to,test_num=10))
    #rows, cols = len(errors[0]),len(errors)
    #erplot = [([0]*cols) for i in range(rows)]
    #for i in range(len(errors)):
    #    plt.plot(errors[i],label=calendar.day_abbr[i])
    his = [0] * (to-1)
    for i in range(len(errors)):
        minVal = min(errors[i])
        his[errors[i].index(minVal)] += 1
    plt.bar(list(range(1,to)),his)
    plt.xlim([0,to])
    plt.legend()
    plt.show()
