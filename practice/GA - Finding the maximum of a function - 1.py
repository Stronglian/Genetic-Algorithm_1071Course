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
        self.crossoverPair = 6 #配對用到組數
        self.crossoverRate = 0.25#交配率
        
        self.mutationRate = 0.5 #突變率
        
        
        #func
        self.FitnessFunc = lambda x : x**2
#        self.CalValue = lambda li : sum([ num for i in range(self.bitNum) num=(2**i)*li[i] ])
        
        #
        self.recordFitnessMax = ['', 0]
    def GenerateBitString(self):
        bitString = ''
        for b in range(self.bitNum):
            bitString += random.choice(['0', '1'])
        return bitString
    def CalBitValue(self, bitString):
        #計算二位元轉十進位
        sumNum = 0
        for i, bit in enumerate(bitString[::-1]):#倒序
            bit = int(bit)
            sumNum += (2**i)*bit
        return sumNum
    def Mutation(self, inputStr):
        #變異
        newStr = ''
        for temp in inputStr:
            ranTemp = random.random()
            if ranTemp > self.mutationRate:
                temp = '0' if temp == '1' else '1'
            newStr+= temp
        return newStr
    def MainFlow(self):
        #儲存空間，String、Fitness
        strArr = np.array([ '' for i in range(self.populationSize)])
        fitnessArr = np.array([ 0 for i in range(self.populationSize)])
        #生成 對應組數
        for i in range(self.populationSize):
            strArr[i] = self.GenerateBitString()
        #Fitness，並記錄
        for i in range(self.populationSize):
            fitnessArr[i] = self.FitnessFunc(strArr[i])
        self.recordFitnessMax = [strArr[fitnessArr.argmax()], fitnessArr[fitnessArr.argmax()]]
        #輪盤法抓人出來配對
        #交配
        #突變
        return
    
    def RouletteWheelSlection(self, inputArr, fitnessArr):
        #靠，不知道要挑幾個，
        #(暫時)依照機率，挑 self.crossoverPair 來挑出來，以weightArr(fitnessArr)來挑選，pairGroup分組，再傳到交配函數處理
        #
        weightArr = fitnessArr.copy().astype(float)
        pairGroup = [-1 for i in range(len(inputArr))] #配對紀錄
        #計算輪盤
        sumWeight = weightArr.sum()
        weightArr /= sumWeight
        #挑選、分組
        pairNum = 0
        while (len(inputArr)-pairGroup.count(-1)) < test.crossoverPair:
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
    def Crossover(inputArr, pairGroup):
        newArr = np.array([ '*'*test.bitNum for i in range(test.populationSize)])
        return newArr
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print('START')
    test = GeneticAlgorithm()
    
    
    #儲存空間，String、Fitness
    strArr = np.array([ '*'*test.bitNum for i in range(test.populationSize)])
    fitnessArr = np.array([ 0 for i in range(test.populationSize)])
    #生成 對應組數
    for i in range(test.populationSize):
        strArr[i] = test.GenerateBitString()
    #Fitness，並記錄
    for i in range(test.populationSize):
        bitVal = test.CalBitValue(strArr[i])
        fitnessArr[i] = test.FitnessFunc(bitVal)
    if fitnessArr[fitnessArr.argmax()] > test.recordFitnessMax[1]:
        test.recordFitnessMax = [strArr[fitnessArr.argmax()], fitnessArr[fitnessArr.argmax()]]
    print(test.recordFitnessMax)
    #輪盤法抓人出來配對
    pairGroup = test.RouletteWheelSlection(strArr, fitnessArr)
    #交配
    #one-ponint
#    #突變
#    for i in range(test.populationSize):
#        strArr[i] = test.Mutation(strArr[i])
        
    
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')