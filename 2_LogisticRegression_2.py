# -*- coding:utf-8 -*-

#对分类问题进行学习拟合

import math
import random
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import time
from loadData import load

#--导入训练数据
data=load(r"F:\SomeMLProjects\ex2data1.txt")
#data前两列为x 最后一列为y 注意要将y化为矩阵形式
x=np.c_[data[:,0],data[:,1]]
y=np.array([data[:,2]]).T

# 以当前时间戳的int为随机数种子
np.random.seed(int(time.time()))

# 对于只有一行的矩阵 np.random.randn输出的也会是个数组中的数组
# theta个数与单个输入x的维度一样 此处1表示列长度
m=np.size(x,1)
theta=np.random.randn(m,1)
#print(theta)

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

class ClassificationLearner(object):
	def __init__(self,alpha,input,output,theta,interation_num,lambda_num):
		self.alpha=alpha
		self.input=input
		self.output=output
		self.theta=theta
		self.inter_num=interation_num
		self.lambda_num=lambda_num
		self.X=self.mapFeature(self.input[:,0],self.input[:,1])
		#self.X=self.input
		self.theta=np.zeros((np.shape(self.X)[1],1))
		#print(self.X)
		#print(self.theta)
		
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))
		#sgm=((z+np.abs(z)))*(1/(1+np.exp(-z)))\
		#	+((np.abs(z)-z))*(1-1/(1+np.exp(z)))
		
	def mapFeature(self,X1,X2):
		temp=np.array([np.ones(np.shape(X1))]).T
		# 用1到6次方的全展开项来拟合
		# range从0开始到5 我们需要i从1到6 j从0到i
		for i in range(2):
			if i!=0:
				for j in range(i+1):
					tj=j+1
					temp=np.c_[temp,np.multiply(np.power(X1,i-j),np.power(X2,j))]
		#print(np.shape(temp))
		return temp
	
	def getCostFunction(self):
		m=len(self.output)
		prediction=self.sigmoid(np.dot(self.X,self.theta))
		
		# 这里！巨坑！！！我服了！！
		# 如果直接写成fixtheta=self.theta 两者会共享内存！！修改fixtheta会导致self.theta也被修改！
		# 	我服了！所以像这样先把内容取出来再用np.array新建！下面求梯度也是这样
		#	您没电 我上门

		fixtheta=np.array([self.theta.T[0]])
		#fixtheta[:,0]=0
		#print(np.shape(np.multiply(y,np.log(prediction))))
		# 这式子有点复杂 带正则化项写在octave里的是
		# J=(-1/m)*sum((y.*log(prediction))+(1-y).*log(1-prediction))
		#	+(lambda/(2*m))*sum(fixtheta.^2);
		
		# 又是一个坑！
		# 这些数学库的log默认都是ln！Octave和numpy都是！
		# 	和数学课上学的并不一样！！
		#print(np.log(1-prediction))
		J=(-1/m)*np.sum(np.multiply(y,np.log(prediction))+np.multiply((1-y),np.log(1-prediction))) \
			+(self.lambda_num/(2*m))*np.sum(np.power(fixtheta,2))
		
		return J

	def gradientDescent(self):			# 梯度下降 天下第一！
		m=len(self.output)
		prediction=self.sigmoid(np.dot(self.X,self.theta))
		
		fixtheta=np.array([self.theta.T[0]])
		#fixtheta=self.theta.T
		#fixtheta[:,0]=0
		
		# 这里octave里写的是
		#grad=(1/m)*sum((prediction-y).*X); fixtheta2=(lambda/m)*theta'; fixtheta2(:,1)=0; grad=grad+fixtheta2;
		grad=(1/m)*np.sum(np.multiply(prediction-y,self.X),axis=0).T+(self.lambda_num/m)*fixtheta
		
		#print(grad.T)
		#print(np.shape(grad))
		#print(np.shape(self.theta))
		# numpy真的有点np 如果ab矩阵相减行列不匹配不报错 居然输出一个大小等于a行数乘b列数的矩阵 我服了
		self.theta=self.theta-self.alpha*grad.T

	def plotFeature(self,theta,X,y,nowinternum):
		plt.clf()
		#绘制样本点
		pos = np.where(y == 1)[0]
		neg = np.where(y == 0)[0]
		X1 = X[pos, 0:2]
		X0 = X[neg, 0:2]
		plt.plot(X1[:, 0], X1[:, 1], 'k+')
		plt.plot(X0[:, 0], X0[:, 1], 'yo')
		plt.xlabel('Microchip Test 1 | '+str(nowinternum))
		plt.ylabel('Microchip Test 2')
		plt.legend(labels=['y = 1', 'y = 0'])
		#绘制决策边界
		#print(np.shape(self.X))
		if np.shape(self.X)[1]>3:
			u = np.linspace(-1, 1.5, 50)
			v = np.linspace(-1, 1.5, 50)
			z = np.zeros((np.size(u),np.size(v)))
			#print(np.shape(z))
			#print(np.shape(theta))
			#print(np.shape(self.mapFeature(u,v)))
			
			
			for i in range(0, np.size(u)):
				for j in range(0, np.size(v)):
					z[i, j] = np.dot(self.mapFeature(u[i], v[j]), theta)
			#print(np.shape(z))
			# print(z)
			plt.contour(u, v, z.T, [0])
		#plt.show()
		else:
			#print(self.X[:,1])
			plot_x = np.array([np.min(self.X[:,1])-2, np.max(self.X[:,1])+2])
			#print(plot_x)
			plot_y = (-1/self.theta[2])*(self.theta[1]*plot_x+self.theta[0])
			#print(plot_y)
			plt.plot(plot_x,plot_y)
			plt.legend(labels=['Admitted', 'Not admitted'])
			plt.axis([30, 100, 30, 100])
		plt.pause(0.001)
		
if __name__=='__main__':
	Learner=ClassificationLearner(0.0014,x,y,theta,100000,0)
	print(Learner.getCostFunction())
	for i in range(600000):
		plt.ion()
		plt.figure(1)
		Learner.gradientDescent()
		J_history.append(Learner.getCostFunction())
		i_history.append(i)
		if i%10000==0 or(i<=100 and i>=50 and i%20==0) or (i<50 and i%10==0):
			Learner.plotFeature(Learner.theta,x,y,i)
	
	plt.pause(5)	
	plt.figure(2)
	plt.plot(i_history,J_history,'k')
	plt.xlabel('i-Interation Times')
	plt.ylabel('J-Cost Value')
	plt.show()
	plt.pause(5)
	print(np.shape(Learner.theta))
	#Learner.plotFeature(Learner.theta,x,y,i)
	
	

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
