from typing import Optional
from fastapi import FastAPI
from parkPredictor import getCurrent, makePrediction
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import json

class Req(BaseModel):
    WEEKDAY: int
    WEEK_SHIFT: int
    POI_ID: int

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
    poi_ID = int(item.POI_ID)
    return getCurrent(poi_ID)

@app.post("/predict")
async def create_item(item: Req):
    index = int(item.WEEKDAY)
    week_shift = int(item.WEEK_SHIFT)
    poi_ID = int(item.POI_ID)
    return makePrediction(index,week_shift,poi_ID)

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, log_level="info")