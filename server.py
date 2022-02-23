from typing import Optional
import sys
from fastapi import FastAPI
from parkPredictor import getCurrent, makePrediction
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import json

class Req(BaseModel):
    WEEKDAY: int
    WEEK_SHIFT: int
    PKLOT_ID: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/current")
async def read_current(item: Req):
    pklot_ID = int(item.PKLOT_ID)
    return getCurrent(pklot_ID)

@app.post("/predict")
async def create_item(item: Req):
    index = int(item.WEEKDAY)
    week_shift = int(item.WEEK_SHIFT)
    pklot_ID = int(item.PKLOT_ID)
    return makePrediction(index,week_shift,pklot_ID)

if __name__ == "__main__":
    prt = 5000
    try:
        if sys.argv[1] is None:
            print("Starting server using default server port: 5000")
        else:
            try:
                prt = int(sys.argv[1])
            except:
                print("ERROR: You must specify an integer value for server port.")
                print("Starting server using default server port: 5000")
    except IndexError:
        print("Starting server using default server port: 5000")
    uvicorn.run("server:app", host="127.0.0.1", port=prt, log_level="info")