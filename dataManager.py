import numpy as np
import torch
from tqdm import trange
class DataManager():
    def __init__(self,mode="train",device = "cuda:0"):
        self.device = device
        self.data = torch.tensor(np.load(f"./data/{mode}.npy"),device=self.device)
        self.windowsSize = 50
        self.pvm = torch.full([self.data.shape[2] - self.windowsSize, 12], 1 / 12,device=self.device)
        self.priceTensor = []
        self.priceRelativeVector = []

    def genPriceTensor(self, startIndex):
        batchIdx = list(range(startIndex, startIndex + self.windowsSize))
        batchData = self.data[:,:,batchIdx]
        priceTensor = torch.zeros(size=batchData.shape,device=self.device)
        finalCloses = batchData[0,:,-1]
        for featureNo in range(3):
            for coinNo in range(11):
                for time in range(self.windowsSize):
                    priceTensor[featureNo][coinNo][time] = batchData[featureNo][coinNo][time]/finalCloses[coinNo]
        return priceTensor

    def genPriceRelativeVector(self,startIndex):
        batchIdx = list(range(startIndex, startIndex + self.windowsSize))
        batchData = self.data[:, :, batchIdx]
        v_t = batchData[0,:,-1]
        v_t_1 = batchData[0,:,-2]
        priceRelativeVector = v_t/v_t_1
        return priceRelativeVector

    def genData(self):
        for i in trange(self.data.shape[-1]-self.windowsSize):
            self.priceTensor.append(self.genPriceTensor(startIndex=i))
            self.priceRelativeVector.append(self.genPriceRelativeVector(startIndex=i))
        return torch.stack(self.priceTensor,dim=0),torch.stack(self.priceRelativeVector,dim=0)

if __name__ == '__main__':
    dm = DataManager(mode="train")
    priceTensor, priceRelativeVector = dm.genData()
    ones = torch.ones(size=(priceRelativeVector.shape[0], 1), device="cuda:0")
    priceRelativeVector = torch.cat([ones, priceRelativeVector], dim=1)
    torch.save(priceTensor, f"./data/trainPriceTensor.torchTensor")
    torch.save(priceRelativeVector, "./data/trainPriceRelativeTensor.torchTensor")
