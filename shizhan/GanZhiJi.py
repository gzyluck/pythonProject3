

# 感知机算法


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
num=100# 样本点
#  x1  x2    y 0/1
#y=1        -1,在下面
x_1=np.random.normal(6,1,size=(num))#  随机数，在6附近产生，产生规律为高斯分布
x_2=np.random.normal(3,1,size=(num))
y=np.ones(num)*(-1)#  一行
#print(y)
c_1=np.array([x_1,x_2,y]) #3行100列
#print(c_1.shape)
# y=0         1  在上面
x_1=np.random.normal(5,1,size=(num))#  随机数，在6附近产生，产生规律为高斯分布
x_2=np.random.normal(6,1,size=(num))
y=np.ones(num)
c_0=np.array([x_1,x_2,y]) #3行100列

c_1=c_1.T   # 100行 3列
c_0=c_0.T

plt.scatter(c_1[:,0],c_1[:,1])
plt.scatter(c_0[:,0],c_0[:,1],marker="+")
plt.show()  #上面两行代码写在一起，，用一个show()可以把点搞到一个图例面

all_data=np.concatenate((c_1,c_0))  #参数是一个元组
#print(all_data)
#print(all_data.shape) #200行 3列
np.random.shuffle(all_data)

train_data_x=all_data[:150,:2] # 150行2列
#print(train_data_x.shape)
train_data_y=all_data[:150,-1].reshape(150,1)# 150行 列
#train_data_y=train_data_y.reshape(1,150)
#print(train_data_y.shape)
test_data_x=all_data[150:,:2]
test_data_y=all_data[150:,-1].reshape(50,1)



w=np.zeros((2,1))
t=1000
k=0
train_data=np.concatenate((train_data_x,train_data_y),axis=1)
#print(train_data.shape)   #(150,3)

#训练模型
for  i in range(t):
    np.random.shuffle(train_data)
    for i in range(len(train_data)):
        pre = np.dot(w.T,(train_data[i][-1]*train_data[i][:2]).reshape(2,1))[0,0]
        if pre<=0:
            w=w+(train_data[i][-1]*train_data[i][:2]).reshape(2,1)



#   y=w1*x1+w2*x2
#   w1*x+w2*y=0
plt.scatter(c_1[:,0],c_1[:,1])
plt.scatter(c_0[:,0],c_0[:,1],marker="+")
x=np.arange(10)  # 根据这个图 大概看出来 X 大概到10 左右
y=-(w[0]*x)/w[1]
plt.plot(x,y)
plt.show()










































