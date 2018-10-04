# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:40:36 2018

@title: Finding the maximum of a function
原有功能都有，只是清掉 Debug 用的Code
"""
"""
reference:
    反轉字串:
        http://yehnan.blogspot.com/2015/04/python.html
    
"""
import numpy as np
import random

class GeneticAlgorithm_maxOfFunc():
    def __init__(self):
        """
        wheelGetDiffPopOnlyTF:控制在取配對的時候可不可以單一字串參與多組配對，if true，輸出(strArr)仍有順序姓，else，持續往後疊加
        """
        #set 
        self.wheelGetDiffPopOnlyTF = False
        #value
        self.bitNum = 5 #位元數
        self.populationSize   = 8 #總字串數
        
        self.tournamentSize   = 2 # 一組配對的字串數
#        self.crossoverPair    = 3  #配對用到組數 #靠，不知道要挑幾個，
        self.crossoverPair    = self.populationSize//self.tournamentSize -1  #配對用到數 
        
        self.crossoverRate    = 0.5 #交配率
        
        self.mutationRate     = 0.2 #突變率
        self.repeatGeneration = 20 # = 世代數量 -1
        
        #func
        self.FitnessFunc = lambda x : x**2
        
        #record
        self.recordFitnessMax = ['', 0, 0]
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
    def CalCorrespondValue(self, bitString):
        """將 x 轉換到對應的值˙"""
        x = self.CalBitValue(bitString)
        return x
    def RouletteWheelSlection(self, inputArr, fitnessArr):
        """(暫時)依照機率，挑 self.crossoverPair*self.tournamentSize 來挑出來，以weightArr(fitnessArr)來挑選，pairGroup分組，再傳到交配函數處理
        也可以不用機率直接用 ranTmp = random.randint(0, sumWeight)做處理，但比較慢。
        不重複取
        """
        weightArr = (fitnessArr.copy().astype(float))+1
        pairGroup = [[] for i in range(len(inputArr))] #配對紀錄
        #計算輪盤 #挑選、分組
        pairNum = 0
        pairNumCount = 0
        while pairNum < self.crossoverPair:
            ranTmp = random.randint(0, weightArr.sum())
            for i in range(len(weightArr)):
                if ranTmp < sum(weightArr[:i+1]):
                    #控制重複取與否
                    if len(pairGroup[i]) != 0 and self.wheelGetDiffPopOnlyTF:
                        weightArr[i] = 0
                        break
                    pairGroup[i].append(pairNum)
                    pairNumCount += 1
                    #已配對計數
                    if pairNumCount == self.tournamentSize:
                        pairNum += 1
                        pairNumCount = 0
                    break
        
        return pairGroup
    def Crossover(self, inputArr, pairGroup):
        """依照交配率看成功與否，在 random 從哪裡交錯
        one-ponint
        """
#        print('Crossover','in-', inputArr)
        if self.wheelGetDiffPopOnlyTF:
            newLi = np.array([ '*'*self.bitNum for i in range(self.populationSize)])
        else:
            newLi = []
        tmpPairLi = np.zeros(self.tournamentSize, dtype=np.int)#暫存要交換的 index 
        for i in range(self.crossoverPair): #第幾對
#            print(i, '---')
            #找配對的
            j = 0
            while j != self.tournamentSize:
                for k, pairNumLi in enumerate(pairGroup):
                    if i in pairNumLi:
                        tmpPairLi[j] = k
                        j += 1
                    if j == self.tournamentSize:
                        break
#            print(tmpPairLi)
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
                if self.wheelGetDiffPopOnlyTF:
                    #如果從頭儲存，無法看變化，所以就按照位置吧；
                    for k in range(self.tournamentSize):
                        newLi[tmpPairLi[k]] = tmpStrLi[k]
                else:
                    #要保留舊有的
                    for k in range(self.tournamentSize):
                        newLi.append(tmpStrLi[k])
            else:
                #沒配對要換掉配對內容變 -1 ，最後統一處理
                for k in range(self.tournamentSize):
                    pairGroup[tmpPairLi[k]].remove(i)
        
#        print('Crossover','pairGroup', pairGroup)
        #處理沒有crossover的
        if self.wheelGetDiffPopOnlyTF:
            for k, pairNumLi in enumerate(pairGroup):
                if len(pairNumLi) == 0:
                    newLi[k] = inputArr[k]
        else:
            for k, pairNumLi in enumerate(pairGroup):
                if len(pairNumLi) == 0:
                    newLi.append(inputArr[k])
        
#        print('Crossover','out-', newLi)
        return np.array(newLi)
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
        #開始世代輪替
        for count in range(self.repeatGeneration): 
#            print(count, '-', strArr)
            #Fitness，並記錄
            for i in range(self.populationSize):
                bitVal = self.CalCorrespondValue(strArr[i])
                fitnessArr[i] = self.FitnessFunc(bitVal)
            tmpArgmax = fitnessArr.argmax()
            if fitnessArr[tmpArgmax] > self.recordFitnessMax[-1]:
                self.recordFitnessMax = [strArr[tmpArgmax], self.CalCorrespondValue(strArr[tmpArgmax]), fitnessArr[tmpArgmax]]
#            print(self.recordFitnessMax)
#            print('after Fitness:',strArr)
            #輪盤法抓人出來配對
            #--更新 配對數
            self.crossoverPair    = len(strArr)//self.tournamentSize -1  #配對用到數 
            pairGroup = self.RouletteWheelSlection(strArr, fitnessArr)
#            print('after RouletteWheelSlection:',strArr)
#            print('pairGroup:', pairGroup)
            #交配
            strArr = self.Crossover(strArr, pairGroup)
#            print('after Crossover:',strArr)
            #突變
            for i in range(self.populationSize):
                strArr[i] = self.Mutation(strArr[i])
#            print('after Mutation:',strArr,'\n\n')
        print('總子代數量:',len(strArr))
        return self.recordFitnessMax
if __name__ == '__main__' :
    import time
    startTime = time.time()
    print('START')
    test = GeneticAlgorithm_maxOfFunc()
    
    print('Final', test.MainFlow())
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')