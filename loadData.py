# -*- coding:utf-8 -*-

import numpy as np
import os

import struct
import numpy as np
import matplotlib.pyplot as plt

# 为了再也不会有人哭泣 我们来封装读入数据的函数
# 普通数据读入就从coursera的txt开始
def load(filepath,delimit=','):
	if os.path.exists(filepath):
		pass
	else:
		filepath=input("当前目录未检测到数据文件 请输入文件所在目录:")+"\\"+filepath
	
	return np.loadtxt(filepath,delimiter=delimit)

# 读取MNIST的数据 每次返回为方便使用 将单幅像素矩阵化为一行的行向量 总共为样本数*图像大小的矩阵
def loadMNISTimage(filepath):
	print("===正在读取图像===")
	fdata=open(filepath,"rb").read()

	if(fdata):
		offset=0
		# 数据头为int形式 对应使用i表示该格式
		fheaderlength='>4i'
		magicnum,imagenum,rownum,colnum=struct.unpack_from(fheaderlength,fdata,offset)

		print("当前文件共有：",imagenum,"\b张",rownum,"\b行",colnum,"\b列的图像")

		# 数据为unsigned byte形式 对应使用B表示该格式
		ilength=">"+str(rownum*colnum)+"B"
		offset+=struct.calcsize(fheaderlength)
		image=[]

		for i in range(imagenum):
			imagebyte=struct.unpack_from(ilength,fdata,offset)

			# 将每个图像转为1*单幅图像大小的矩阵
			# 每次向image列表中添加一个行向量列表 最后再将整个列表np矩阵化
			#oneimage=(np.array(imagebyte).reshape(1,rownum*colnum))[0]
			oneimage=list(imagebyte)
			image.append(oneimage)

			offset+=struct.calcsize(ilength)

		print("===图像读取完成===\n")

		return np.array(image)
	
	else:
		print("未找到指定数据文件 请确认文件路径是否正确")
		return -1
		exit()

# 读取MNIST的标签 总共为样本数*1的矩阵
def loadMNISTtag(filepath):
	print("===正在读取标记===")
	ftag=open(filepath,"rb").read()
	
	if(ftag):
		offset=0
		# 数据头为int形式 对应使用i表示该格式
		fheaderlength='>2i'
		magicnum,tagnum=struct.unpack_from(fheaderlength,ftag,offset)

		print("当前文件共有：",tagnum,"\b个类别标记")

		# 数据为unsigned byte形式 对应使用B表示该格式
		tlength=">1B"
		offset+=struct.calcsize(fheaderlength)
		tag=[]

		for i in range(tagnum):
			tagbyte=struct.unpack_from(tlength,ftag,offset)

			# 每次添加一个tag到list 形成1*样本数的list 最后返回时矩阵化并转置
			ontag=list(tagbyte)

			tag.append(ontag)

			offset+=struct.calcsize(tlength)

		print("===标记读取完成===\n")
		return np.array(tag)
	
	else:
		print("未找到指定数据文件 请确认文件路径是否正确")
		return -1
		exit()

# 用来一次性返回图像和对应标签的字典 就像mat文件那样
def loadMNIST(imagefilePath,tagfilePath):
	image=loadMNISTimage(imagefilePath)
	tag=loadMNISTtag(tagfilePath)

	return {'X':image,'y':tag}
	
if __name__=='__main__':
	#print(Load(r"F:\SomeMLProjects\ex2data2.txt"))
	#image=loadMNISTimage(r"F:\SomeMLProjects\HandWriting\train-images.idx3-ubyte")
	#tag=loadMNISTtag(r"F:\SomeMLProjects\HandWriting\train-labels.idx1-ubyte")

	data=loadMNIST(r"F:\SomeMLProjects\HandWriting\train-images.idx3-ubyte",\
		r"F:\SomeMLProjects\HandWriting\train-labels.idx1-ubyte")
	
