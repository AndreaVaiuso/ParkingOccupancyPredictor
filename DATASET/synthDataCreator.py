import os
import random
from telnetlib import SGA
from csvtools import csv_open, CsvDataFrame

def alterValue(val):
    s = True
    if random.random() > 0.5: s = False
    if s:
        val += (random.random() / 8)
        if val >= 1: return 1
        else: return val
    else:
        val -= (random.random() / 8)
        if val <= 0: return 0
        else: return val

available_parks = [5,7,13,14,22,34,35,36,37,45,46,48,53]
his_orig = csv_open("history.csv")

for pknum in available_parks:
    his_temp = his_orig.getDataFrame()
    path = "PARKING_LOTS/"+str(pknum)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    for row in his_temp:
        row["BLOCK_ID_COUNT"] = str(pknum)
        row["TOTAL_OCCUPIED_RATIO"] = str(alterValue(float(row["TOTAL_OCCUPIED_RATIO"])))
    df = CsvDataFrame(his_temp)
    df.csvOut(path+"/history.csv")
    

    