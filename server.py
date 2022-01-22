from typing import Optional
from fastapi import FastAPI
from csvtools import csv_open
from main import prepare, predict, SAMPLING_TIME
from utilities import secToTime
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import json

class Req(BaseModel):
    WEEK_DAY: float
    WEEK_SHIFT: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/current")
def read_current():
    ds = csv_open("DATASET/current_day.csv")
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

@app.post("/predict")
async def create_item(item: Req):
    global models, trends, look_back
    index = int(item.WEEK_DAY)
    week_shift = int(item.WEEK_SHIFT)
    if index not in range(7):
        return json.dumps({"status": "failed"})
    print("Request: ",index," Week_shift: ",week_shift)
    model,trend = prepare(index) 
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

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, log_level="info")