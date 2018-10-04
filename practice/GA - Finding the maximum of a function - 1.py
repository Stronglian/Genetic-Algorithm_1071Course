# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:40:36 2018

@title: Finding the maximum of a function
"""
"""
reference:
    反轉字串:
        http://yehnan.blogspot.com/2015/04/python.html
    
"""
import numpy as np
import random

class GeneticAlgorithm():
    def __init__(self):
        #value
        self.bitNum = 5 #位元數
        self.populationSize = 8 #總組數
        
        self.tournamentSize = 2 # 一組配對的組數
        self.crossoverPair = 3 #配對用到組數#靠，不知道要挑幾個，
        self.crossoverRate = 0.5#交配率
        
        self.mutationRate = 0.8 #突變率
        self.repeatGeneration = 50 #世代交換
        
        
        #func
        self.FitnessFunc = lambda x : x**2
#        self.CalValue = lambda li : sum([ num for i in range(self.bitNum) num=(2**i)*li[i] ])
        
        #
        self.recordFitnessMax = ['', 0]
    def GenerateBitString(self):
        """"""
        bitString = ''
        for b in range(self.bitNum):
            bitString += random.choice(['0', '1'])
        return bitString
    def CalBitValue(self, bitString):
        """計算二位元轉十進位"""
        sumNum = 0
        for i, bit in enumerate(bitString[::-1]):#倒序
            bit = int(bit)
            sumNum += (2**i)*bit
        return sumNum
    
    def RouletteWheelSlection(self, inputArr, fitnessArr):
        """(暫時)依照機率，挑 self.crossoverPair*self.tournamentSize 來挑出來，以weightArr(fitnessArr)來挑選，pairGroup分組，再傳到交配函數處理
        """
        weightArr = fitnessArr.copy().astype(float)
        pairGroup = [-1 for i in range(len(inputArr))] #配對紀錄
        #計算輪盤
        sumWeight = weightArr.sum()
        weightArr /= sumWeight
        #挑選、分組
        pairNum = 0
        while (len(inputArr)-pairGroup.count(-1)) < self.crossoverPair * self.tournamentSize:
            ranTmp = random.random()
#            ranTmp = random.randint(0, sumWeight) 
            for i in range(len(weightArr)):
                if ranTmp < sum(weightArr[:i+1]):
                    if pairGroup[i] != -1:
                        #代表已經有分組了
                        break
                    pairGroup[i] = pairNum
                    if pairGroup.count(pairNum) == self.tournamentSize:
                        pairNum += 1
                    break
        
        return pairGroup
    def Crossover(self, inputArr, pairGroup):
        """依照交配率看成功與否，在 random 從哪裡交錯
        one-ponint
        """
#        print('Crossover','in-', inputArr)
        newArr = np.array([ '*'*self.bitNum for i in range(self.populationSize)])
        tmpPairLi = np.zeros(self.tournamentSize, dtype=np.int)#暫存要交換的 index 
        for i in range(self.crossoverPair): #第幾對
            #找配對的
            j = 0
            for k, pairNum in enumerate(pairGroup):
                if pairNum == i:
                    tmpPairLi[j] = k
                    j += 1
                if j == self.tournamentSize:
                    break
            #配對與否
            if random.random() > self.crossoverRate:
                #one-point 交換點
                crossoverPonint = random.randint(0, self.bitNum)
                tmpStrLi = ['' for i in range(self.tournamentSize)] #暫存交換玩的String
                for j in range( self.bitNum):
                    if j < crossoverPonint:
                        for k in range(self.tournamentSize):
                            tmpStrLi[k] += inputArr[tmpPairLi[k]][j]
                    else: #後段交換
                        for k in range(self.tournamentSize):#輪換
                            tmpStrLi[k] += inputArr[tmpPairLi[k+1 if k+1 < self.tournamentSize else 0]][j]
                #如果從頭儲存，無法看變化，所以就按照位置吧
                for k in range(self.tournamentSize):
                    newArr[tmpPairLi[k]] = tmpStrLi[k]
            else:
                #沒配對要換掉配對內容變 -1 
                for k in range(self.tournamentSize):
                    pairGroup[tmpPairLi[k]] = -1
#                #OR 直接換
#                for k in range(self.tournamentSize):
#                    newArr[tmpPairLi[k]] = inputArr[tmpPairLi[k]]
        
#        print('Crossover','pairGroup', pairGroup)
        #處理沒有crossover的
        for k, pairNum in enumerate(pairGroup):
            if pairNum == -1:
                newArr[k] = inputArr[k]
        
#        print('Crossover','out-', newArr)
        return newArr
    def Mutation(self, inputStr):
        """變異"""
        newStr = ''
        for temp in inputStr:
            ranTemp = random.random()
            if ranTemp > self.mutationRate:
                temp = '0' if temp == '1' else '1'
            newStr+= temp
        return newStr
    def MainFlow(self):
        """"""
        #儲存空間，String、Fitness
        strArr = np.array([ '*'*self.bitNum for i in range(self.populationSize)])
        fitnessArr = np.array([ 0 for i in range(self.populationSize)])
        #生成 對應組數
        for i in range(self.populationSize):
            strArr[i] = self.GenerateBitString()
        for count in range(self.repeatGeneration):
#            print(count, '-', strArr)
            #Fitness，並記錄
            for i in range(self.populationSize):
                bitVal = self.CalBitValue(strArr[i])
                fitnessArr[i] = self.FitnessFunc(bitVal)
            if fitnessArr[fitnessArr.argmax()] > self.recordFitnessMax[1]:
                self.recordFitnessMax = [strArr[fitnessArr.argmax()], fitnessArr[fitnessArr.argmax()]]
    #        print(self.recordFitnessMax)
    #        print('after Fitness:',strArr)
            #輪盤法抓人出來配對
            pairGroup = self.RouletteWheelSlection(strArr, fitnessArr)
    #        print('after RouletteWheelSlection:',strArr)
    #        print('pairGroup:', pairGroup)
            #交配
            strArr = self.Crossover(strArr, pairGroup)
    #        print('after Crossover:',strArr)
            #突變
            for i in range(self.populationSize):
                strArr[i] = self.Mutation(strArr[i])
    #        print('after Mutation:',strArr,'\n\n')
                
        print('Final', self.recordFitnessMax)
        return self.recordFitnessMax
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print('START')
    test = GeneticAlgorithm()
    
    test.MainFlow()
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')