# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 02:51:31 2018

@title: Optimization of a simple function of one variable
"""
"""
修改:
    1.
     -1. RouletteWheelSlection 遇到 fitfunc 會生負數
      - 針對最小樹小於零的數列，直接加最小值的絕對值
     -2. fitness 儲存格改成 float
     -3. 是 隨機數字小於變異率(交配) 才會變異(交配)
    2. 字串運算
     - 提升效率，字串、過去未知函數
     - 補充備註
     - random vs np.random
    *. (下一站)，'101' 改成 [1,0,1]
    *. (下一站)，跑到趨緩在停
     - self.repeatGeneration FREE
"""
"""
reference:
    
"""
import numpy as np
import random
#random.seed('111')
class GeneticAlgorithm():
    def __init__(self):
        """
        __wheelGetDiffPopOnlyTF__:控制在取配對的時候可不可以單一字串參與多組配對，if true，輸出(strArr)仍有順序姓，else，持續往後疊加
        """
        #set 
        self.__wheelGetDiffPopOnlyTF__ = True
        #value
        #題目指定
        self.bitNum = 22 #位元數 #因為要精確到六位數，所以 2**21 < (self.domainUpperbound-self.domainLowerbound)*(10**6) <2**22
        self.populationSize   = 50 #總字串數
        self.crossoverRate    = 0.25 #交配率
        self.mutationRate     = 0.01 #突變率
        #自訂或相應的
        self.tournamentSize   = 2 # 一組配對的字串數
        self.crossoverPair    = self.populationSize//self.tournamentSize -1  #配對用到數 
        self.repeatGeneration = 50 # = 世代數量 -1
        
        #題目指定
        self.domainUpperbound = 2.0
        self.domainLowerbound = -1.0
        #function
        self.FitnessFunc = lambda x : (x)*np.sin(31.4*x)+1.0
        
        #record
        self.recordFitnessMax = ['', 0, 0.0] #string, x, fitness
    def GenerateBitString(self):
        """生成二進位字串"""
        bitString = ''
        for b in range(self.bitNum):
            bitString += random.choice(['0', '1'])
        return bitString
    def CalBitValue(self, bitString):
        """計算二位元轉十進位"""
#        sumNum = 0
#        for i, bit in enumerate(bitString[::-1]):#倒序
#            bit = int(bit)
#            sumNum += (2**i)*bit
        return int(bitString, 2)#sumNum
    def CalCorrespondValue(self, bitString):
        """將 x 轉換到對應的值˙"""
        x_ = self.CalBitValue(bitString)
        x = self.domainLowerbound + x_*(self.domainUpperbound - self.domainLowerbound)/( 2**self.bitNum -1)
        return x
    def RouletteWheelSlection(self, inputArr, fitnessArr):
        """(暫時)依照機率，挑 self.crossoverPair*self.tournamentSize 來挑出來，以weightArr(fitnessArr)來挑選，pairGroup分組，再傳到交配函數處理
        也可以不用機率直接用 ranTmp = random.randint(0, sumWeight)做處理，但比較慢。
        不重複取
        """
        weightArr = (fitnessArr.copy().astype(float)) +1 #處理0的問題
        pairGroup = [[] for i in range(len(inputArr))] #配對紀錄
        #計算輪盤 #挑選、分組
        pairNum = 0 #已有的配對數量
        pairNumCount = 0 #該配對有幾個成員(bitString)
        if weightArr.min() < 0: #若有值小於0就把所有值加上它的絕對值。
            weightArr += np.absolute(weightArr.min())
        while pairNum < self.crossoverPair:
            ranTmp = random.uniform(0, weightArr.sum())
            for i in range(len(weightArr)): #輪盤每一格的判斷
                if ranTmp < sum(weightArr[:i+1]):
                    #控制重複取與否
                    if len(pairGroup[i]) != 0 and self.__wheelGetDiffPopOnlyTF__:
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
        if self.__wheelGetDiffPopOnlyTF__:
            newLi = np.array([ '*'*self.bitNum for i in range(self.populationSize)])
        else:
            newLi = []
        tmpPairLi = np.zeros(self.tournamentSize, dtype=np.int)#暫存要交換的 index 
        for i in range(self.crossoverPair): #第幾對
#            print(i, '---')
            #找配對的
            pairNumCount = 0
            while pairNumCount != self.tournamentSize:
                for k, pairNumLi in enumerate(pairGroup):
                    if i in pairNumLi:
                        tmpPairLi[pairNumCount] = k
                        pairNumCount += 1
                    if pairNumCount == self.tournamentSize:
                        break
#            print(tmpPairLi)
            #配對與否
            if random.random() < self.crossoverRate:
                #one-point 交換點
                crossoverPonint = random.randint(1, self.bitNum) #= 0 的時候不就沒配對了
                tmpStrLi = ['' for i in range(self.tournamentSize)] #暫存交換的String
                #Crossover
                for k in range(self.tournamentSize):
                    tmpStrLi[k] += inputArr[tmpPairLi[k]][0:crossoverPonint]
#                    else: #後段交換
                    #輪換，為了維持可以多組交配
                    tmpStrLi[k] += inputArr[tmpPairLi[k+1 if k+1 < self.tournamentSize else 0]][crossoverPonint:]
                #儲存
                if self.__wheelGetDiffPopOnlyTF__:
                    #如果從頭儲存，無法看變化，所以就按照位置吧；
                    for k in range(self.tournamentSize):
                        newLi[tmpPairLi[k]] = tmpStrLi[k]
                else:
                    #要保留舊有的，而且同位置可能有兩組，所以直接尾綴。
                    for k in range(self.tournamentSize):
                        newLi.append(tmpStrLi[k])
            else:
                #沒配對要踢掉配對內容，最後統一處理
                for k in range(self.tournamentSize):
                    pairGroup[tmpPairLi[k]].remove(i)
#        print('Crossover','pairGroup', pairGroup)
        #處理沒有crossover的
        if self.__wheelGetDiffPopOnlyTF__:
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
            if ranTemp < self.mutationRate:
                temp = '0' if temp == '1' else '1'
            newStr+= temp
        return newStr
    def MainFlow(self):
        """"""
        #儲存空間，String、Fitness
        strArr = np.array([ '*'*self.bitNum for i in range(self.populationSize)])
        fitnessArr = np.zeros(self.populationSize, dtype = np.float)
        #生成 對應組數
        for i in range(self.populationSize):
            strArr[i] = self.GenerateBitString()
#        print('initialGroup----\n', strArr)
        #開始世代輪替
        for count in range(self.repeatGeneration): 
#            print(count, '-', strArr)
            #Fitness，並記錄
            for i in range(self.populationSize):
                corspondVal = self.CalCorrespondValue(strArr[i])
                fitnessArr[i] = self.FitnessFunc(corspondVal)
            tmpArgmax = fitnessArr.argmax()
            if fitnessArr[tmpArgmax] > self.recordFitnessMax[-1]:
                self.recordFitnessMax = [strArr[tmpArgmax], self.CalCorrespondValue(strArr[tmpArgmax]), fitnessArr[tmpArgmax]]
#            print(self.recordFitnessMax)
#            print('after Fitness:',strArr)
            #輪盤法抓人出來配對
            #--更新 配對數
            self.crossoverPair = len(strArr)//self.tournamentSize -1  #配對用到數 
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
    test = GeneticAlgorithm()
    ans = test.MainFlow()
    print('Final::', 'String:',ans[0], 'Value(x):', round(ans[1],5), 'Fitness:', round(ans[2], 5))
#    print(test.Mutation('101'))
    
    
    endTime = time.time()
    print('\n\n\nEND,', 'It takes', endTime-startTime ,'sec.')
