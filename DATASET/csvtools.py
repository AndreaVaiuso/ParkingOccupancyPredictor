from multiprocessing.sharedctypes import Value
from typing import Type
import datetime

def getArrayFromString(input:str, typ:Type=int):
    rtlist = []
    s1 = input.split("[")
    s2 = s1[1].split("]")
    lst = s2[0].split(",")
    for el in lst:
        rtlist.append(typ(el))
    return rtlist

def arrayToString(input:list):
    string = "["
    delim = ""
    typ = type(input[0])
    if typ == str:
        delim = '"'
    for i in range(len(input)):
        if i<len(input) - 1: string += delim + str(input[i]) + delim + ","
        else: string += delim + str(input[i]) + delim + "]"
    return string

def convertToType(elem:str,prototype):
    typ = type(prototype)
    if typ == list:
        typ == type(prototype[0])
        return getArrayFromString(elem,typ)
    return typ(elem)


class CsvDataFrame():
    csvFile = None
    sep = ""
    def __init__(self, csvFile:dict, sep:str=";"):
        self.csvFile = csvFile
        self.sep = sep
    def setSep(self,sep:str):
        self.sep = sep
    def getDataFrame(self):
        return self.csvFile
    def getHead(self):
        return list(self.csvFile[0])
    def setDataType(self,dataPrototype:list):
        h = self.getHead()
        if len(h) != len(dataPrototype):
            raise ValueError("Inappropriate argument value")
        for row in self.getDataFrame():
            for i in range(len(h)):
                row[h[i]] = convertToType(row[h[i]],dataPrototype[i])
        return self.getDataFrame()
    def removeColumn(self,columnName:str):
        df = self.getDataFrame()
        h = self.getHead()
        index = -1
        for i in range(len(h)):
            if h[i] == columnName:
                index = i
        if index == -1:
            raise ValueError("Column " + columnName + " does not exist")
        for line in df:
            del line[columnName]
        return df
    def csvOut(self,file):
        content = ""
        h = self.getHead()
        for i in range(len(h)):
            if i < len(h) - 1: content += h[i] + self.sep
            else: content += h[i] + "\n"
        df = self.getDataFrame()
        dflen = len(df)
        ct = 0
        for row in df:
            for i in range(len(h)):
                if i < len(h) - 1: content += str(row[h[i]]) + self.sep
            else: 
                if ct < dflen - 1 : content += str(row[h[i]]) + "\n"
                else: content += str(row[h[i]])
            ct += 1
        with open(file,mode="w+") as csvfile:
            csvfile.write(content)
        print(f"Dataset created ({(ct+1)} lines): {file}")
    def getColumn(self,columnName:str):
        col = []
        df = self.getDataFrame()
        for line in df:
            col.append(line[columnName])
        return col

def csv_open(file:str,sep:str=";"):
    csvFile = []
    lines = []
    with open(file,"r",encoding='utf-8-sig') as csvfile:
        lines = csvfile.readlines()
    h = lines[0]
    head = h.strip().split(sep)
    first = True
    i = 0
    for line in lines:
        if first:
            first = False
            continue
        row = {}
        x = line.strip().split(sep)
        for i in range(len(head)):
            row[head[i]] = x[i]
        csvFile.append(row)
        i+=1
    return CsvDataFrame(csvFile,sep=sep)