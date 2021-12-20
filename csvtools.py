from utilities import printProgressBar

def csv_open(file:str,sep:str=";"):
    csvFile = []
    lines = []
    with open(file,"r") as csvfile:
        lines = csvfile.readlines()
    h = lines[0]
    head = h.split(sep)
    first = True
    i = 0
    for line in lines:
        if first:
            first = False
            continue
        row = {}
        x = line.split(sep)
        for i in range(len(head)-1):
            row[head[i]] = x[i]
        csvFile.append(row)
        i+=1
    return csvFile
