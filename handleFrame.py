import pandas as pd
import os

def readData(csv_file="data.csv"):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file, index_col=0)
    df = pd.DataFrame(columns=[f"KQ[{i}]" for i in range(27)])
    df.to_csv(csv_file)
    return df
def readDataL2(csv_file="dataTrain.csv"):
    return pd.read_csv(csv_file, index_col=0).to_numpy()

def saveData(data, csv_file="data.csv"):
    data.to_csv(csv_file)
def sortData(data):
    data.index = pd.to_datetime(data.index, format='%d-%m-%Y')   
    data = data.sort_index()
    data.index = data.index.strftime('%d-%m-%Y')
    return data
def readUniqueData(csv_file="data.csv"):
    return pd.read_csv(csv_file, index_col=0).drop_duplicates()

def convertToL2(uniqueData = readUniqueData()):    
    return uniqueData.apply(lambda col: col.map(lambda x: x % 100))

