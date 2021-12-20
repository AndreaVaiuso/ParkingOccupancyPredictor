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

SECONDS_IN_WEEK = 604800
SECONDS_IN_DAY = 86400

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return numpy.array(dataX), numpy.array(dataY)
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
                print("Reached maximum week limit. Data truncated at value: "+str(j))
                stop = True
                break
        trend[week].reverse()
        if stop: break
    trend.reverse()
    last_day_trend.reverse()
    return trend, last_day_trend, last_day

def getDataset(trend):
    data = []
    for i in range(len(trend)):
        for val in trend[i]:
            data.append(val)
    return data

def convertToColumn(data):
	column = []
	for i in range(len(data)):
		column.append([data[i]])
	return numpy.array(column)

def splitDs(data, splitVal = 0.7):
	n = len(data)
	train_ds = data[0:int(n * splitVal)]
	val_ds = data[int(n * splitVal):n]
	return train_ds, val_ds

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
df = ct.csv_open("DATASET/input.csv",sep=";")
weeks_number = 13
history_trend, current_day_trend, current_weekday = getOldTrend(df,weeks_number)
current_day_len = len(current_day_trend)
dataset = getDataset(history_trend)
train, test = splitDs(dataset)
dataset = numpy.transpose(numpy.array(dataset))
train = convertToColumn(train)
test = convertToColumn(test)
dataset = convertToColumn(dataset)
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(32, input_shape=(1, look_back)))
model.add(Dense(1))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3,mode='min')
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset,label="Ground Truth")
plt.plot(trainPredictPlot,label="Prediction (train)")
plt.plot(testPredictPlot, label="Prediction (test)")
plt.legend()
plt.show()