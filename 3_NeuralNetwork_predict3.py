# -*- coding:utf-8 -*-

#对分类问题进行学习拟合

import math
import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from scipy.io import savemat
import threading

from loadData import loadMNIST

#--导入训练数据
#data=loadmat(r"F:\SomeMLProjects\ex4data1.mat")
data=loadMNIST(r"F:\SomeMLProjects\HandWriting\t10k-images.idx3-ubyte",\
		r"F:\SomeMLProjects\HandWriting\t10k-labels.idx1-ubyte")

Theta=loadmat(r"F:\SomeMLProjects\LearnedThetas.mat")
Theta1=Theta['Theta1']
Theta2=Theta['Theta2']
# 得到的输入x纵向为不同样本 横向为单个样本图案
# y纵向为不同样本的分类 但y本身需要转化为对于不同类别的0/1向量 
#	如对类别1就是类别1为1 其余为0的向量 针对手写数字有10类 按习惯划分为1-9和0对应的10
#	但python中就对应0到9就可以了 但这里mat给的数据就是0对应10 NMDWSM
# 	注意 truey的列需要对应后面计算出来的prediciton
x=data['X']
y=data['y']

truey=np.zeros((np.shape(y)[0],10))
yi=0
for ynum in y:
	truey[yi][ynum]=1
	yi+=1

# 以当前时间戳的int为随机数种子
np.random.seed(int(time.time()))

# sigmoid函数及使用它的结果能得到的sigmoid的导数
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
	
def sigmoidGradient(z):
	g=sigmoid(z)
	return np.multiply(g,(1-g))

def predict(X,Theta1,Theta2,mode=0):
	# 这是对于整个神经网络而言的前向传播 mode=0预测单个样本类别 mode=1计算所有样本而言的总正确率
	m=np.shape(X)[0]
	a1=np.hstack((np.ones((m,1)),X))

	z2=np.dot(a1,Theta1.T)
	g2=sigmoid(z2)
	m2=np.shape(g2)[0]
	a2=np.hstack((np.ones((m2,1)),g2))

	z3=np.dot(a2,Theta2.T)
	g3=sigmoid(z3)
	a3=g3

	#从a3中找出最大值取值位置作为预测值 需要转置 单个样本无所谓
	if(mode==0):
		pos=np.argmax(a3,axis=1)
		#pos=np.array([list(pos)]).T

		return pos
	else:
		pos=np.argmax(a3,axis=1)
		pos=np.array([list(pos)]).T
		result=(pos==y)+0
		Accuracy=np.sum(result)/np.size(result)
		return Accuracy
	

# 正式开始

if __name__=='__main__':
	quit=0

	print("The full accuracy is:",predict(x,Theta1,Theta2,mode=1)*100,"\b%")

	# 以随机一个样本作为输入进行预测
	figure1=plt.figure(1)
	while(quit<=100):
		index=np.random.randint(np.shape(x)[0])

		predition=predict(np.array([list(x[index])]),Theta1,Theta2)
		print("I guess it might be:",predition)
		print("It actually is:",y[index],"\n")
		plt.clf()
		plt.imshow(np.reshape(x[index],(28,28)))
		plt.draw()
		plt.pause(1.5)
		quit+=1
	exit(0)
	
	