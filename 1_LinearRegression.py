# -*- coding:utf-8 -*-

#对单变量函数进行学习拟合
#如y=kx+b y=sinx
#值得注意的是 虽然y=ax^2+bx+c看起来也是单变量 但在机器学习里应把x^2视为另一个变量进行学习(如x为x1，x^2为x2)

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

#--导入训练数据
#	这里选择直接生成
init=[]
initnum=50
for num in range(initnum):
	init.append([1])
X=np.array(init)

init=[]
for num in range(initnum):
	init.append([(num+random.random())/10])
tlist=np.array(init)
X=np.c_[X,tlist]

init=[]
for num in range(initnum):
	init.append([(math.sin(num/10)+random.random())/10])
tlist=np.array(init)
X=np.c_[X,tlist]
y=tlist

'''
X=np.array([[1,1,1],
			[1,2,4],
			[1,3,9],
			[1,4,16]])

y=np.array([[1],
			[4],
			[9],
			[16]])
'''
#以当前时间戳的int为随机数种子
np.random.seed(int(time.time()))

#对于只有一行的矩阵 np.random.randn输出的也会是个数组中的数组 
#总之这里只取第一个元素
#theta=np.random.randn(2,1)
theta=np.array([[0],
				[-1],
				[0]])
print(theta)

testX=[]
testY=[]
J_history=[]
i_history=[]

#--使用的模型函数 拒绝重复代码
def predictFunction(input,theta):
	return (theta[0]+theta[1]*input+theta[2]*math.sin(input))[0]

#--object表示要继承的类 这里没有特别的类就继承object类
#--self在实例中表示创建的实例本身 指代自己 
#		并且在定义对象内函数时 使用self即可访问该类初始化中定义的数据
#		无需再以内函数参数形式传入 相当于类内部构成某种小空间 初始化的变量相当于
#		该空间的全局变量
#--利用初始化将创建实例时的外界输入赋值给自身属性
#		alpha-学习率 input-输入数据矩阵 output-真实的结果数据矩阵
#		theta-要学习的权重矩阵 interation_num-迭代轮数

class OneLayerLearner(object):
	def __init__(self,alpha,input,output,theta,interation_num):
		self.alpha=alpha
		self.input=input
		self.output=output
		self.theta=theta
		self.inter_num=interation_num
		
	def getCostFunction(self):
		m=len(self.output)
		prediction=np.dot(self.input,self.theta)
		#print(prediction-y)
		sqrError=(prediction-y)*(prediction-y)
		J=(1/(2*m))*np.sum(sqrError)
		
		return J

	def gradientDescent(self):
		m=len(self.output)
		
		plt.ion()
		plt.figure(1)
		plt.xlim((0,10))
		plt.ylim((0,10))
		
		for i in range(self.inter_num):
			#print("theta:",self.theta)
			prediction=np.dot(self.input,self.theta)
			#print("prediction:",prediction)
			Error=prediction-y
			#print("Error:",Error)
			
			# 按照公式 
			#		如theta0 对应于 所有样本的误差与样本中第0个参数值相乘的和
			#		所以应该是将样本误差矩阵与样本参数矩阵对应相乘后 取每列的和 
			#		最终得到一行 再转置
			derivative=(1/m)*np.sum(np.multiply(Error,self.input),axis=0)
			derivative=np.array([list(derivative)]).T
			
			self.theta=self.theta-self.alpha*derivative
			
			#print("theta:",self.theta)
			if i%1000==0 or i<=100:
				plt.clf()
				plt.plot(self.input[:,1],self.output,'b-o')
				tX=[]
				tY=[]
				for k in range(initnum):
					tX.append(k)
					tY.append(predictFunction(k,self.theta))
				plt.plot(tX,tY,'r-*')
				plt.xlabel(i)
				plt.pause(0.01)
			
			J=self.getCostFunction()
			J_history.append(J)
			i_history.append(i)
		
		print("theta:",self.theta)
	

if __name__=='__main__':
	Learner=OneLayerLearner(0.1,X,y,theta,100000)
	Learner.gradientDescent()
	
	plt.figure(1)
	plt.plot(X[:,1],y,'b-o')
	for k in range(10):
		testX.append(k)
		testY.append(predictFunction(k,Learner.theta))
	plt.plot(testX,testY,'r-*')
	plt.xlim((0,10))
	plt.ylim((0,10))
	
	plt.figure(2)
	plt.plot(i_history,J_history,'k-*')
	plt.show()
	

#-----以下是在学习时使用octave的时候写的代码 涉及到很细节的东西-------
# 计算误差J 按照一般的X和theta的存储方式 一般使用X*theta
#prediction=X*theta;		
#sqrError=(prediction-y).^2;

#J=(1/(2*m))*sum(sqrError);

# 梯度下降 时刻注意矩阵的长宽方向！
#prediction=X*theta;
#Error=prediction-y;

#dev=(1/m)*sum(Error.*X);
#dev=dev';			

#theta=theta-alpha*dev;
