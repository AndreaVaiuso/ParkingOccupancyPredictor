import csvtools as ct
import datetime as dt
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from utilities import printErr

SECONDS_IN_WEEK = 604800
SECONDS_IN_DAY = 86400

class DataWindow:
    data = None
    index = None
    def __init__(self,data):
        self.data = data
        self.index = len(data)-1
    def getLast(self):
        return self.data[self.index]
    def setIndex(self, index):
        self.index = index
    def incrementIndex(self):
        self.index += 1
    def getData(self):
        return self.data
    def getWindow(self):
        window = []
        for i in range(0,self.index+1):
            window.append(self.data[i])
        return window

def getCurrentTrend(data_file,label_column="TOTAL_OCCUPIED_RATIO"):
    trend = []
    for line in data_file:
        trend.append(float(line[label_column]))
    trend_window = DataWindow(trend)
    return trend_window

def getOldTrend(data_frame,weeks_number=3,samplingTime=3600,label_column="TOTAL_OCCUPIED_RATIO"):
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
        i -= int(SECONDS_IN_WEEK / samplingTime) 
        for j in range(i,i-int(SECONDS_IN_DAY/samplingTime),-1):
            try:
                trend[week].append(float(data_frame[j][label_column]))
            except IndexError:
                printErr("Reached maximum week limit. Data truncated at value: "+str(j))
                stop = True
                break
        trend[week].reverse()
        if stop: break
    trend.reverse()
    last_day_trend.reverse()
    return trend, last_day_trend, last_day

def getOldTrend2(data_frame,weeks_backward=3,backward=24,samplingTime=3600,label_column="TOTAL_OCCUPIED_RATIO"):
    i = -1
    trend = []
    for week in range(weeks_backward):
        for j in range(i,i-backward,-1):
            trend[week].append(data_frame[j][label_column])
        i -= int(SECONDS_IN_WEEK / samplingTime)
    return trend

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def generateWeights(size,firstElementGain=1):
    weights = []
    for x in range(size):
        val = -math.atan(x)+(math.pi/2)
        if x == 0: val *= firstElementGain
        weights.append(val)
    return normalize(weights,min(weights),max(weights))

def plotf1(history_trend,current_day_trend,limit=3):
    y = range(1,25)
    plt.plot(y,current_day_trend,label="Current day trend")
    for i in range(limit):
        plt.plot(y,history_trend[i],label="Week "+str(i))
    plt.legend()
    plt.show()

def loadModel():
    lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(units=1)
    ])
    return lstm_model

def getDataset(trend,splitVal = 0.7):
    data = []
    for i in range(len(trend)):
        for val in trend[i]:
            data.append(val)
    return splitDs(data,splitVal)

def splitDs(data,splitVal):
    n = len(data)
    train_ds = data[0:int(n * splitVal)]
    val_ds = data[int(n * splitVal):n]
    return train_ds, val_ds

if __name__ == "__main__":
    current_day_trend_window_csv_file = ct.csv_open("DATASET/to_predict.csv",sep=";")
    current_day_trend_window = getCurrentTrend(current_day_trend_window_csv_file)
    current_day_trend_window.setIndex(2)
    weeks_number=14
    df = ct.csv_open("DATASET/input.csv",sep=";")
    history_trend, current_day_trend, current_weekday = getOldTrend(df,weeks_number)
    current_day_len = len(current_day_trend)
    ##plotf1(history_trend,current_day_trend)
    ##weights = generateWeights(weeks_number)
    ds_train, ds_valid = getDataset(history_trend)
    model = loadModel()
    MAX_EPOCHS = 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    ##TODO 
    history = model.fit(ds_train, epochs=MAX_EPOCHS, validation_data=ds_valid, callbacks=[early_stopping])
