import numpy as np
import torch
import torch.nn as nn
from dataManager import DataManager
from getPV import pvCaculator
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import trange
class portFolioModel(nn.Module):
    def __init__(self,windowSize=50):
        super(portFolioModel, self).__init__()
        self.windowsSize = windowSize
        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=48,kernel_size=(1,3))
        self.relu1 = nn.ReLU()
        self.Conv2 = nn.Conv2d(in_channels=48,out_channels=20,kernel_size=(1,48))
        self.relu2 = nn.ReLU()
        self.Conv3 = nn.Conv2d(in_channels=21,out_channels=1,kernel_size=(1,1))
        self.sm = nn.Softmax(dim=1)
        self.priceTensor = torch.load("./data/testPriceTensor.torchTensor")
        self.priceRelativeTensor = torch.load("./data/testPriceRelativeTensor.torchTensor")
        self.pvm = np.full([self.priceTensor.shape[0], 12], 1 / 12)

    def forward(self,inputFeature,lastW,):
        conv1 = self.relu1(self.Conv1(inputFeature))
        conv2 = self.relu2(self.Conv2(conv1))
        # [50,20,11,1]을[20,50,11,1] 로 변환
        conv2 = conv2.permute(1,0,2,3)
        #[50,11]
        lastW = lastW[:,1:]
        # [50,11]을 [1,50,11,1]로 만들어야 [20,50,11,1]과 콘캣 해서 [21,50,11,1] 만듬
        lastW = torch.unsqueeze(lastW, axis=0)
        lastW = torch.unsqueeze(lastW, axis=3)
        cat = torch.cat([conv2,lastW])
        cat = cat.permute(1,0,2,3)
        # cat[:,-1,:,:]의 값이 초기화로 잘 들어감
        conv3 = self.Conv3(cat).squeeze()
        if len(conv3.shape) ==1:
            conv3 = torch.unsqueeze(conv3,dim=0)
        btcBias = torch.zeros((inputFeature.shape[0], 1),device="cuda:0")
        output = self.sm(torch.cat([btcBias,conv3],dim=1))
        return output

import random
if __name__ == '__main__':
    # training
    batchSize = 50
    model = portFolioModel().to("cuda:0")
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    indexList = range(1,2407)
    startIndex = 1
    for epoch in range(1000):
        indexList = list(range(2407-50))
        random.shuffle(indexList)
        for startIndex in indexList:
            batchIndex = list(range(startIndex,startIndex+batchSize))
            tensor = model.priceTensor[batchIndex]
            priceRelativeTensor = model.priceRelativeTensor[batchIndex]
            lastW =  torch.tensor(torch.from_numpy(model.pvm[np.array(batchIndex)-1]),device="cuda:0",dtype=torch.float)
            currentW = model.forward(inputFeature=tensor,lastW=lastW)
            model.pvm[batchIndex] = currentW.detach().cpu()
            value = priceRelativeTensor * lastW
            wprime = value / torch.sum(value, dim=1).unsqueeze(-1)
            mu = 0.999
            optimizer.zero_grad()
            logMean = -torch.mean(torch.log((priceRelativeTensor*currentW)))
            loss = logMean
            loss.backward()
            optimizer.step()
            # print(epoch,"startIndex: ",startIndex,loss)

            with torch.no_grad():
                model.eval()
                priceTensor = torch.load("./data/testPriceTensor.torchTensor")
                priceRelativeTensor = torch.load("./data/testPriceRelativeTensor.torchTensor")
                evalLastW = torch.zeros(size=[1,12],device="cuda:0",dtype=torch.float32)
                evalLastW[0][0] = 1
                testArr = []
                for testIndex in range(priceTensor.shape[0]):
                    tensorTest = torch.unsqueeze(priceTensor[testIndex],dim=0)
                    currentWTest = model.forward(inputFeature=tensorTest,lastW=evalLastW)
                    wPrimeTest = pvCaculator.getWPrime(yt=priceRelativeTensor[testIndex], lastW=evalLastW[0])
                    # muTest = pvCaculator.getMuT(wPrime=wPrimeTest, wt=currentWTest[0])
                    testArr.append((torch.dot(priceRelativeTensor[batchIndex][0], lastW[0]) * mu).item())
                print("cumProd: ",np.cumprod(np.array(testArr)))

