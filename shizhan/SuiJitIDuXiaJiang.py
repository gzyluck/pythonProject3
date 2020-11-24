
''''
#numpy的数组类型
#numpy是Python中科学计算的基础包，它是一个Python库，提供多维数组对象，
#各种派生对象，如掩码数组和矩阵以及用于数组快速操作的各种例程。
array=[1,2,4]
print(array,"hhh")
print(type(array))
nparray=np.array([6,5,4,3,2])  # nparray,这个变量是矩阵。
nparray2=np.array([1])
print(nparray)
print(type(nparray))
print(nparray+1)
print("============ndarray类型================")
list_=[1,2,3,4]
array_=np.array(list_)
print(array_.dtype) #查看数组里面是什么数据类型
print(array_.size)  # 矩阵大小
print(array_.ndim) # 矩阵维度
list2_=[1,2,3,4.0]   #有一个是浮点型，，则这个矩阵整体表现为 浮点型
array2_=np.array(list2_)
print(array2_.dtype)
print("============索引、切片================")
#  ndarray的索引和切片跟PY一样的
list_=[1,2,3,4]
array_=np.array(list_)
print(array_[0])
print(array_[-1])
print(array_[0:3:2])  #  qie切片
print("============复杂数组================")
erweishuzu =np.array([[1,2,3],[4,5,6],[7,8,9]])
print(erweishuzu.size)
print(erweishuzu.shape)
print(erweishuzu.ndim)
erweishuzu[1,1]=10
print(erweishuzu)
print("按列取值，，，冒号+逗号+索引值",erweishuzu[:,1])
print("按2列",erweishuzu[:,0:2])
print("按2列",erweishuzu[0:2:,0:2])

=========================================


list_=np.arange(0,60,10) # 步长为10
print(list_)
list2=np.array([1,0,3,0,6,0],dtype=bool)  #  将其改为布尔类型
print(list2)
print(list_[list2])  #  为TURE时 ，输出，否则不输出。。。找出非0数。。。
random_array=np.random.rand(10) # 产生随机数组
print(random_array)
random_array2=np.random.randint(0,1000)# 产生随机数,在给定的范围内
print(random_array2)

=========================================


random_juzhen=np.random.randint(1,5,size=(3,3,3)) # 一个矩阵 大小 3行3列。在1到10之间的数
print(random_juzhen)
print(np.sum(random_juzhen))    # 矩阵的值都加起来，算和
print(np.sum(random_juzhen,axis=0))#   列相加(2为数组)
print(np.sum(random_juzhen,axis=1))#     行相加（2维数组）
print(random_juzhen.ndim)   #  变量名字，维度ndim
=============================

random_2=np.random.randint(1,5,size=(2,2))
print(random_2)
print(np.prod(random_2)) # 求乘机
print(np.prod(random_2,axis=0))
print(random_2.max())
print(random_2.max(axis=0))
=========================================

#排序。。。arg  前缀 是索引。。。。 axis 确定是第几个维度。。sort默认按照最后一个维度排序
random_3=np.random.randint(1,5,size=(2,5)) #  两行五列
print(random_3)
print(np.sort(random_3,axis=0))
print(np.argsort(random_3,axis=0))
print("===============")
print(np.sort(random_3,axis=1))
print(np.argsort(random_3,axis=1))
=========================================

#  矩阵的转置
random_array2=np.random.randint(1,10,size=(2,5))
print(random_array2)
print(random_array2.T)  #  矩阵转置
=========================================

#  矩阵的拼接
random_array3=np.random.randint(10,size=(5))
print(random_array3)
random_array4=np.random.randint(10,size=(5))
print(random_array4)
array=np.concatenate(random_array3,random_array4)#     这行代码哟错误 array=np.concatenate(random_array3,random_array4)
print(array,axis=1)  #     File "<__array_function__ internals>", line 6, in concatenate
 #       TypeError: only integer scalar arrays can be converted to a scalar index
     
=========================================

a=np.arange(0,2)
print(a)
b=np.arange(3,5)
print(b)
print(a*b) # 这个不是矩阵相乘。 下面这个是矩阵相乘
print("---------------------")
x=np.arange(0,6).reshape(2,3)
print(x)
y=np.arange(0,6).reshape(3,2)
print(y)
print(np.dot(x,y))
=======================================
# *基础感知：线性拟合数据...10行代码感知什么是机器学习
# *20180815

import numpy as np
#原始数据
X=[ 1 ,2  ,3 ,4 ,5 ,6]
Y=[ 2.6 ,3.4 ,4.7 ,5.5 ,6.47 ,7.8]

#用一次多项式拟合，相当于线性拟合
z1 = np.polyfit(X, Y, 1)#x与Y为需要拟合的数据，n为需要拟合的函数的阶数（次数）；例子中的n为1.   z1为拟合
p1 = np.poly1d(z1)  # p1是多项式
print (z1)  #[ 1.          1.49333333]
print (p1)  # 1 x + 1.493

=========================================
'''

#随机梯度下降
import numpy as np
import  matplotlib.pyplot as plt
x=np.arange(0,50)
print(x)
# (-5,5)  制造噪声
np.random.seed(1)
randomarray=-(np.random.random(50)*2-1)*5   # (-5,5)  制造噪声
y=2*x+randomarray


# plt.scatter（）绘制散点图
plt.scatter(x,y)
plt.show()
x=x.reshape(50,1)     #  50行一列
#print(x.shape) #x的类型是多维数组
y=y.reshape(50,1)
#print(y)
all_data=np.concatenate((x,y),axis=1)  #拼接  按照第二个维度,,表示对应行(维度)的数组进行拼接
print(all_data.ndim)         #  all_data是2维数组对象
#划分数据集     随机取(切片操作) 独立同分布[打乱数据  取钱40个]
np.random.shuffle(all_data)# 随机打乱数据集
#print(all_data)
train_data=all_data[0:40]
test_data =all_data[40:50]
#  算法实现
#超参数
lr=0.001# 学习率
N=100
errorrate=300 #我们希望的 损失和不能大于200.限定为200
#参数
theta=np.random.rand()   #  初始化函数参数[产生一个随机数在0-1之间] 贼\c他
#   ===========
num=1
theta_list=[]
loss_list=[]
while True:
    # 重新排序
    np.random.shuffle(train_data)#打乱训练集
    for n in range(N):
        #取随机值
        rand_int=np.random.randint(0,40)
        rand_x  =train_data[rand_int][0]
        rand_y  =train_data[rand_int][1]
        #计算梯度 ，，事先已经算好了，按照平方差作为损失 ，，求偏导数计算梯度
        grad=(rand_x*theta-rand_y)*rand_x
        #geng更新theta
        theta=theta-lr*grad
    #计算更新后（的损失，，套到所有的X中 ）错误率errorrate
    x=train_data[:,0]
    y=train_data[:,1]
    loss=np.sum(0.5*(theta*x-y)**2)# 损失之和
    print("number :%d, theta: %f, loss :%f"%(num,theta,loss))
    num=num+1
    theta_list.append(theta)
    loss_list.append(loss)
    if loss<errorrate:
        break
# theta 的值变化图
plt.plot(range(len(theta_list)),theta_list)
plt.show()
# 损失 的值变化图
plt.plot(range(len(loss_list)),loss_list)
plt.show()


















































































































































































































































































