import requests,json
from datetime import datetime
import pytz
import pandas as pd
from tqdm import tqdm
import numpy as np
minute = 60
halfAnHour = minute * 30
hour = minute * 60
day = hour * 24
week = day * 7
month = day * 30
year = day * 365
class polonexDataGenerator():
    def __init__(self):
        self.trainStart = datetime.strptime("2014-11-01 00:00", "%Y-%m-%d %H:%M").astimezone(pytz.utc).timestamp()
        self.trainEnd = datetime.strptime("2016-09-07 04:00", "%Y-%m-%d %H:%M").astimezone(pytz.utc).timestamp()
        self.testStart = self.trainEnd
        self.testEnd = datetime.strptime("2016-10-28 08:00", "%Y-%m-%d %H:%M").astimezone(pytz.utc).timestamp()
        self.selectionStart = self.testStart - month


    def getCurrencyPair(self):
        url = "https://poloniex.com/public?command=returnTicker"
        return json.loads(requests.get(url=url).text)

    def getChart(self, pair, start, end, period):
        url = f"https://poloniex.com/public?command=returnChartData&currencyPair={pair}&start={start}&end={end}&period={period}"
        return json.loads(requests.get(url=url).text)

    def getVolume(self):
        pairList = []
        for pair, data in tqdm(self.getCurrencyPair().items()):
            volumeSum = 0
            if str(pair).startswith("BTC_"):
                data = self.getChart(pair=pair, start=self.selectionStart, end=self.testStart, period=day)
                for item in data:
                    volumeSum = volumeSum + item["volume"]
                pairList.append({"pair":pair,"volume":volumeSum})
            elif str(pair).endswith("_BTC"):
                data = self.getChart(pair=pair, start=self.selectionStart, end=self.testStart, period=day)
                for item in data:
                    volumeSum = volumeSum + item["quoteVolume"]
                pairList.append({"pair":pair,"volume":volumeSum})
        df = pd.DataFrame(pairList)
        df.to_csv("./data/pairToVolume.csv",index=None)

    def getAllCHL(self,start,end,mode="train"):
        featureData = []
        self.coinList = pd.read_csv("./data/pairToVolume.csv").sort_values(by="volume", ascending=False).head(11)
        for coinPair in tqdm(self.coinList["pair"]):
            chartList = []
            current = start
            while current+month<end:
                chart = self.getChart(pair=coinPair,start=current,end=current+month,period=halfAnHour)
                chartList.extend(chart)
                current += month
            chart = self.getChart(pair=coinPair, start=current, end=end, period=halfAnHour)
            chartList.extend(chart)
            for item in chartList:
                if str(coinPair).startswith("BTC_"):
                    featureData.append({"coinPair":coinPair,"date":item['date'],"close":item['close'],"high":item['high'],"low":item['low']})
                if str(coinPair).endswith("_BTC"):
                    try:
                        featureData.append({"coinPair": coinPair, "date": item['date'], "close": 1/item['close'], "high": 1/item['high'], "low": 1/item['low']})
                    except Exception as ex:
                        print(item)
        df = pd.DataFrame(featureData)
        df.to_csv(f"./data/{mode}CHLData.csv",index=None)

    def genFeature(self,mode="train"):
        df = pd.read_csv(f"./data/{mode}CHLData.csv")
        df = df[df["date"] > 0]
        data = np.full([3, 11, len(df["date"].unique())], np.NAN, dtype=np.float32)
        for featureNo, featureName in enumerate(["close", "high", "low"]):
            for coinPairNo,coinPairName in enumerate(df["coinPair"].unique()):
                temp = df[df["coinPair"]==coinPairName][[featureName,"date"]]
                if mode=="train":
                    timeIndex = np.array((temp["date"] - self.trainStart) / 1800, dtype=int)
                else:
                    timeIndex = np.array((temp["date"] - self.testStart) / 1800, dtype=int)
                data[featureNo][coinPairNo][timeIndex] = temp[featureName].values
        for fea in range(3):
            for coin in range(11):
                for timeIndex in range(data.shape[-1]-1,-1,-1):
                    if np.isnan(data[fea][coin][timeIndex]):
                        data[fea][coin][timeIndex] = data[fea][coin][timeIndex+1]
        assert not np.any(np.isnan(data))
        print(data.shape)
        np.save(file=f"./data/{mode}.npy",arr=data)

if __name__ == '__main__':
    pdg = polonexDataGenerator()
    # pdg.getAllCHL(start=pdg.testStart,end=pdg.testEnd,mode="test")
    pdg.genFeature(mode="test")