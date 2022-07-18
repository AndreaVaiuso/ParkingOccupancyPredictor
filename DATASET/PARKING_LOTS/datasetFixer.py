import pandas as pd
import os
import sys
from utilities import bcolors

def printloc(df,loc):
    for i in range(loc-5,loc+5):
        if i == loc:
            print(bcolors.OKGREEN + str(df.iloc[i]) + bcolors.ENDC)
        else: print(df.iloc[i])


h_list = ['00:00:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00', '04:00:00', '04:30:00', '05:00:00', '05:30:00', '06:00:00', '06:30:00', '07:00:00', '07:30:00', '08:00:00', '08:30:00', '09:00:00', '09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00', '12:30:00', '13:00:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00', '15:30:00', '16:00:00', '16:30:00', '17:00:00', '17:30:00', '18:00:00', '18:30:00', '19:00:00', '19:30:00', '20:00:00', '20:30:00', '21:00:00', '21:30:00', '22:00:00', '22:30:00', '23:00:00', '23:30:00']


filelist = os.listdir(".")
f_2 = []
for file in filelist:
    if file.endswith(".csv"):
        f_2.append(file)
filelist = f_2

totalerror = {}

for file in filelist:
    print("Opening",file)
    df = pd.read_csv(file)
    j = 0
    errcount = 0
    for j in range(0,len(df)):
        if j > len(df)-1:
            break
        i = j%48
        if df.iloc[j]["hour"] != h_list[i]:
            errcount += 1
            date_error = df.iloc[j]["date"]
            print("Error at index:",j,":",df.iloc[j]["hour"], "should be", h_list[i])
            offset = 0
            for x in range(j,j+48):
                if df.iloc[x]["hour"] == h_list[i] and df.iloc[x]["date"] == date_error:
                    printloc(df,j)
                    for del_index in range(j,j+offset):
                        df.drop(j,inplace=True)
                        print(bcolors.FAIL + "Deleted row: " + str(del_index) + bcolors.ENDC)
                        df = df.reset_index()
                        df = df.drop(columns=["index"])
                    printloc(df,j)
                else: offset += 1
    totalerror[file] = str(errcount)
    df.to_csv(file,index=False)
print()
for key in totalerror:
    print(key,":",totalerror[key])