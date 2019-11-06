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

from loadData import loadMNIST

#--导入训练数据及标签
#data=loadmat(r"F:\SomeMLProjects\ex4data1.mat")
data=loadMNIST(r"F:\SomeMLProjects\HandWriting\train-images.idx3-ubyte",\
		r"F:\SomeMLProjects\HandWriting\train-labels.idx1-ubyte")
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

# 对于只有一行的矩阵 np.random.randn输出的也会是个数组中的数组
# theta个数与单个输入x的维度一样 此处1表示列长度
m=np.size(x,1)

J_history=[]
i_history=[]

#--object表示要继承的类 这里没有特别的类就继承object类
#--self在实例中表示创建的实例本身 指代自己 
#		并且在定义对象内函数时 使用self即可访问该类初始化中定义的数据
#		无需再以内函数参数形式传入 相当于类内部构成某种小空间 初始化的变量相当于
#		该空间的全局变量
#--利用初始化将创建实例时的外界输入赋值给自身属性
#		alpha-学习率 in-输入数据矩阵 out-真是所给的结果数据矩阵
#		theta-要学习的权重矩阵 interation_num-迭代轮数

class NeuralNetwork_ThreeLayers(object):
	def __init__(self,alpha,indata,outdata,theta1,theta2,lambda_num):
		self.alpha=alpha
		self.lambda_num=lambda_num
		
		self.X=indata
		self.y=outdata
		
		imageSize=np.shape(self.X[0])[0]

		#self.Theta1=theta1		
		#self.Theta2=theta2
		self.Theta1=self.initialThetas(imageSize,25)
		self.Theta2=self.initialThetas(25,10)
		
		self.Theta1_grad=np.zeros(np.shape(self.Theta1))
		self.Theta2_grad=np.zeros(np.shape(self.Theta2))

	# sigmoid函数及使用它的结果能得到的sigmoid的导数
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))
		
	def sigmoidGradient(self,z):
		g=self.sigmoid(z)
		return np.multiply(g,(1-g))
	
	# 必须以随机数初始化权重消除权重矩阵的相关性
	def initialThetas(self,L_in,L_out):
		epsilon_init=np.sqrt(6/(L_in+L_out))
		return np.random.rand(L_out,L_in+1)*2*epsilon_init-epsilon_init

	def feedforwardPropagation(self,X,Thetas):
		# 这是对于整个神经网络而言的前向传播 每次可对X所代表的所有输入样本进行计算得到所有样本的预测输出
		#	本例是只有一层隐藏层的三层全连接神经网络 两段Theta大小分别为25*401和10*26 所以输出为m*10
		# 每层都要添加偏置单元 numpy中使用hstack做到横向合并

		Theta1=np.reshape(Thetas[0:np.size(self.Theta1)],np.shape(self.Theta1),order='F')
		Theta2=np.reshape(Thetas[np.size(self.Theta1):],np.shape(self.Theta2),order='F')

		m=np.shape(X)[0]
		a1=np.hstack((np.ones((m,1)),X))

		#z2=np.dot(a1,self.Theta1.T)
		z2=np.dot(a1,Theta1.T)
		g2=self.sigmoid(z2)
		m2=np.shape(g2)[0]
		a2=np.hstack((np.ones((m2,1)),g2))

		#z3=np.dot(a2,self.Theta2.T)
		z3=np.dot(a2,Theta2.T)
		g3=self.sigmoid(z3)
		a3=g3

		return a1,a2,a3,z2

	def backforwardPropagation(self,Thetas,start_time):
		# 这是对于整个神经网络而言的反向传播 每次针对单个样本计算 最后用于梯度下降调整权值
		m=np.shape(self.X)[0]
		
		Theta1=np.reshape(Thetas[0:np.size(self.Theta1)],np.shape(self.Theta1),order='F')
		Theta2=np.reshape(Thetas[np.size(self.Theta1):],np.shape(self.Theta2),order='F')
		#J=self.getCostFunction(Thetas)
		
		for i in range(m):
			tempx=np.array([list(self.X[i])])
			a1,a2,a3,z2=self.feedforwardPropagation(tempx,Thetas)
			# delta为各层误差 输出层误差此处采用直接相减
			# 	注意！一个样本的预测只用减去它对应的那行y的trueY！！！！
			delta3=a3-self.y[i][:]
			# 隐藏层由反向传播梯度得到 不对偏置单元计算误差
			#delta2=np.multiply(np.dot(delta3,self.Theta2).T[1:].T,self.sigmoidGradient(z2))
			delta2=np.multiply(np.dot(delta3,Theta2).T[1:].T,self.sigmoidGradient(z2))

			self.Theta1_grad=self.Theta1_grad+np.dot(delta2.T,a1)
			self.Theta2_grad=self.Theta2_grad+np.dot(delta3.T,a2)

		# 计算正则项 不对偏置单元的权重学习修改 也即每行第一列不用计算
		fixTheta1=np.array(Theta1)
		fixTheta2=np.array(Theta2)
		#fixTheta1=np.array(self.Theta1)
		#fixTheta2=np.array(self.Theta2)

		fixTheta1[:,0]=0
		fixTheta2[:,0]=0
		#print(self.lambda_num)
		self.Theta1_grad=(1/m)*(self.Theta1_grad+self.lambda_num*fixTheta1)
		self.Theta2_grad=(1/m)*(self.Theta2_grad+self.lambda_num*fixTheta2)

		# 把几层的Theta梯度写为一列向量
		Theta_grads=np.reshape(self.Theta1_grad,(-1,1),order='F')
		Theta_grads=np.vstack((Theta_grads,np.reshape(self.Theta2_grad,(-1,1),order='F')))

		#要使用fmin_cg的话 就要返回一个行向量数组！
		return Theta_grads.T[0]


	def getCostFunction(self,Thetas,start_time):
		fTname=r"F:\SomeMLProjects\LearnedThetas.mat"
		Theta1=np.reshape(Thetas[0:np.size(self.Theta1)],np.shape(self.Theta1),order='F')
		Theta2=np.reshape(Thetas[np.size(self.Theta1):],np.shape(self.Theta2),order='F')
		m=np.shape(self.X)[0]

		# 以a3作为预测输出
		prediction=self.feedforwardPropagation(self.X,Thetas)[2]
		# octave中表达式为
		#	J=(-1/m)*sum(sum(trueY.*log(prediction)+(1-trueY).*log(1-prediction)));
		J=(-1/m)*(np.sum(np.sum(np.multiply(self.y,np.log(prediction))+\
			np.multiply((1-self.y),np.log(1-prediction)))))

		if((time.time()-start_time)%5<(m/5000)):
			print("Now the cost is:",J,"| It took",time.time()-start_time,"s to get us here")
		if((time.time()-start_time)%60<(m/5000)):
			savemat(fTname,{'Theta1':Theta1,'Theta2':Theta2})
			print(" || Automatically saved Theta1 and Theta2 || ")


		# 计算正则项 不对偏置单元的权重学习修改
		fixTheta1=np.array(Theta1)
		fixTheta2=np.array(Theta2)
		#fixTheta1=np.array(self.Theta1)
		#fixTheta2=np.array(self.Theta2)

		fixTheta1[:,0]=0
		fixTheta2[:,0]=0
		# 	(lambda/(2*m))*(sum(sum(fixedTheta1.^2))+sum(sum(fixedTheta2.^2)));
		Jr=(self.lambda_num/(2*m))*(np.sum(np.sum(np.power(fixTheta1,2)))+\
			np.sum(np.sum(np.power(fixTheta2,2))))
		return (J+Jr)


	def gradientDescent(self):			# 梯度下降 天下第一！（但在这里好慢……）
		self.backforwardPropagation()
		self.Theta1=self.Theta1-self.alpha*self.Theta1_grad
		self.Theta2=self.Theta2-self.alpha*self.Theta2_grad
		
		
	def optimize(self):					# fmin_cg 天下第二！（是86！86上山了！）
		# 先把几层的Theta写为一列向量
		Thetas=np.reshape(self.Theta1,(-1,1),order='F')
		Thetas=np.vstack((Thetas,np.reshape(self.Theta2,(-1,1),order='F')))
		#print(Thetas)
		print("====开始最小化====")
		start_time=time.time()
		result=opt.fmin_cg(self.getCostFunction,Thetas,args=(start_time,)\
			,fprime=self.backforwardPropagation,disp=True)
		#result=opt.minimize(self.backforwardPropagation,Thetas,method='CG')
		print(result)
		return result

# 用于生成一个小数据集来确认神经网络确实算对了梯度 生成时直接添加偏置项
def debugInitialVectors(tout,tin):
	totalsize=tout*(tin+1)
	# 和octave里的对应 想用个检查还真麻烦
	return np.reshape(np.sin(np.array(list(range(totalsize)))+1)/10,(tout,tin+1),order='F')

# 确认梯度计算是否正确
def checkGradients():
	inputlayer=3
	hiddenlayer=5
	outputlayer=3
	tm=5
	testTheta1=debugInitialVectors(hiddenlayer,inputlayer)
	testTheta2=debugInitialVectors(outputlayer,hiddenlayer)
	testX=debugInitialVectors(tm,inputlayer-1)
	testY=np.array([[2,3,1,2,3]]).T
	#print(testX)
	#print(testY)
	ty=np.zeros((np.shape(testY)[0],3))
	yi=0
	for ynum in testY:
		if ynum==10:
			ty[yi][9]=1
		else:
			ty[yi][ynum-1]=1
		yi+=1
	#print(ty)
	testNeuralNetwork=NeuralNetwork_ThreeLayers(alpha,testX,ty,testTheta1,testTheta2,interation_num,3)
	testNeuralNetwork.backforwardPropagation()
	#print(np.reshape(testNeuralNetwork.Theta1_grad,(-1,1),order='F'))
	print(testNeuralNetwork.Theta1_grad)
	
# 正式开始

if __name__=='__main__':
	alpha=0.01
	interation_num=150000
	lambda_num=1

	# 继承学习结果
	theta=loadmat(r"F:\SomeMLProjects\LearnedThetas.mat")
	theta1=theta['Theta1']
	theta2=theta['Theta2']

	# 已确认 可以注释掉了
	#checkGradients()

	starttime=time.time()
	fJ=open(r"F:\SomeMLProjects\CostHistory.txt","w+",encoding="utf-8")
	fJ.close()
	fTname=r"F:\SomeMLProjects\LearnedThetas.mat"

	NeuralNetwork=NeuralNetwork_ThreeLayers(alpha,x,truey,theta1,theta2,lambda_num)
	NeuralNetwork.optimize()

	print("Now the cost is:",NeuralNetwork.getCostFunction(),"| It took",time.time()-starttime,"s to get us here")
	savemat(fTname,{'Theta1':NeuralNetwork.Theta1,'Theta2':NeuralNetwork.Theta2})
	print(" || Saved final Theta1 and Theta2 ||")
	'''
	for times in range(interation_num):
		NeuralNetwork.gradientDescent()
		if times%1000==0 or times==10:
			Jnow=NeuralNetwork.getCostFunction()
			print(times,"| Now the cost is:",Jnow,"| It took",time.time()-starttime,"s to get us here")
			fJ=open(r"F:\SomeMLProjects\CostHistory.txt","a",encoding="utf-8")
			fJ.write(str(Jnow)+'\n')
			fJ.close()
		if times%10000==0:
			savemat(fTname,{'Theta1':NeuralNetwork.Theta1,'Theta2':NeuralNetwork.Theta2})

	print("Now the cost is:",NeuralNetwork.getCostFunction())
	print("It cost",time.time()-starttime,"s")
	'''
	fJ.close()
