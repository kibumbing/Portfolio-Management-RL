import torch
import numpy as np

class pvCaculator:
    @staticmethod
    def getWPrime(yt,lastW):
        value = yt * lastW
        wPrime = value / torch.sum(value)
        return wPrime

    @staticmethod
    def getMuT(wPrime,wt):
        commission_rate = 0.0025
        mu0 = torch.ones(size=[50,11],device="cuda:0")
        mu1 = torch.ones(size=[50,11],device="cuda:0") - 2 * commission_rate + commission_rate ** 2
        while torch.all(abs(mu1 - mu0)) > 1e-10:
            mu0 = mu1
            print(f"m0: {mu0}")
            # mu1 = (1 - commission_rate * wPrime[:,0] - (2 * commission_rate - commission_rate ** 2) *
            #        torch.sum(torch.maximum(wPrime[:,1:] - mu1 * wt[:,1:], torch.tensor(0,device="cuda:0",dtype=torch.float32)))) / (1 - commission_rate * wt[0])
            mu1 = (1 - commission_rate * wPrime[:,0] - (2 * commission_rate - commission_rate ** 2) *
                   torch.sum(torch.relu(wPrime[:,1:] - mu1 * wt[:,1:]))) / (1 - commission_rate * wt[:,0])
            print(mu1.shape)

        return mu1
