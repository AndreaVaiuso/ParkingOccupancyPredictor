import sys
from fastapi import FastAPI
from parkPredictor import getCurrent, makePrediction, prepare
import csvtools as ct
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, BaseSettings
import json
import re
from utilities import bcolors

class EnVar(BaseSettings):
    ds: dict = {}
    def getDataSet(self,pklot_id):
        if pklot_id not in self.ds:
            self.ds[pklot_id] = ct.csv_open("DATASET/PARKING_LOTS/"+str(pklot_id)+"/history.csv").getDataFrame()
        return self.ds[pklot_id]


class Req(BaseModel):
    WEEKDAY: int
    WEEK_SHIFT: int
    PKLOT_ID: int

app = FastAPI()
enVar = EnVar()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/api/smartep_ia/occupation/current")
async def read_current(item: Req):
    pklot_ID = int(item.PKLOT_ID)
    return getCurrent(pklot_ID)

@app.post("/api/smartep_ia/occupation/predict")
async def create_item(item: Req):
    index = int(item.WEEKDAY)
    week_shift = int(item.WEEK_SHIFT)
    pklot_ID = int(item.PKLOT_ID)
    try:
        df = enVar.getDataSet(pklot_ID)
    except FileNotFoundError:
        return {"status":"error","msg":"Data frame not found for ID: "+str(pklot_ID)}
    return makePrediction(df,index,week_shift,pklot_ID)

def warmup():
    aviable_pklots = [5,7,13,14,22,24,34,35,36,37,45,46,48,53]
    for pklot_ID in aviable_pklots:
        try:
            df = enVar.getDataSet(pklot_ID)
            for index in range(7):
                prepare(df,index,pklot_ID) 
        except FileNotFoundError:
            continue

if __name__ == "__main__":

    prt = 5000
    hst = "127.0.0.1"
    p = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    try:
        if sys.argv[1] is None:
            print("Usage: " + sys.argv[0] + " [address] [port]")
            print("Starting server using default address:",hst)
        else:
            temp = str(sys.argv[1])
            if p.match(temp):
                hst = temp
                print("Starting server using address:",hst)
            else:
                print(bcolors.FAIL+"ERROR: You must specify a valid address (like 127.0.0.1)"+bcolors.ENDC)
                print(bcolors.WARNING+"Starting server using default address: "+str(hst)+bcolors.ENDC)
        if sys.argv[2] is None:
            print(bcolors.WARNING+"Starting server using default server port: "+prt+bcolors.ENDC)
        else:
            try:
                prt = int(sys.argv[2])
                print("Starting server using port:",prt)
            except:
                print(bcolors.FAIL+"ERROR: You must specify an integer value for server port."+bcolors.ENDC)
                print(bcolors.WARNING+"Starting server using default server port: "+str(prt)+bcolors.ENDC)
    except IndexError:
        print(bcolors.WARNING+"Usage: " + sys.argv[0] + " [address] [port]"+bcolors.ENDC)
        print("Starting server using default server port:",prt)
    uvicorn.run("server:app", host=hst, port=prt, log_level="info")
