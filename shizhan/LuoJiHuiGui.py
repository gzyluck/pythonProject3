
# 逻辑回归实战

import numpy as np
import matplotlib.pyplot as plt
np.random.seed()
num=100# 样本点
#  x1  x2    y 0/1
#y=1        -1,在下面
x_1=np.random.normal(6,1,size=(num))#  随机数，在6附近产生，产生规律为高斯分布
x_2=np.random.normal(3,1,size=(num))
y=np.ones(num)
c_1=np.array([x_1,x_2,y]) #3行100列
#print(c_1.shape)
# y=0         1  在上面
x_1=np.random.normal(5,1,size=(num))#  随机数，在6附近产生，产生规律为高斯分布
x_2=np.random.normal(6,1,size=(num))
y=np.zeros(num)
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
print(train_data_x.shape)
train_data_y=all_data[:150,-1]# 150行 列
#train_data_y=train_data_y.reshape(1,150)
#print(train_data_y.shape)
test_data_x=all_data[150:,:2]
test_data_y=all_data[150:,-1]

# y=w1*x1+w2*x2    w  x  都是向量可以写为 W*X，，，，要理解
w=np.random.rand(2,1)
#print(w)  # 2行1列

#   y=w1*x1+w2*x2
#   w1*x+w2*y=0
plt.scatter(c_1[:,0],c_1[:,1])
plt.scatter(c_0[:,0],c_0[:,1],marker="+")
x=np.arange(10)  # 根据这个图 大概看出来 X 大概到10 左右
y=-(w[0]*x)/w[1]
plt.plot(x,y)
'''
#单条线：
plot([x], y, [fmt], data=None, **kwargs)
#多条线一起画
plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)'''
plt.show()


#定义损失函数 交叉熵
def  cross_entory(y,y_hat):
    return  -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

#y_hat =sigmoid(w*x)
def sigmoid(z):
    return 1./(1+np.exp(-z))

#注意看用到的矩阵的形状，以方便运算不会出错
print(w.shape)
print(train_data_x.shape)

lr=0.001
loss_list=[]
for i in range(1000):
    #计算损失loss
    y_hat=sigmoid(np.dot(w.T,train_data_x.T))
    loss=cross_entory(train_data_y,y_hat)
    #计算梯度
    grad=-np.mean((train_data_x*(train_data_y-y_hat).T),axis=0)
    #更新
    w=w-(lr*grad).reshape(2,1)
    loss_list.append(loss)
    if i%100==0:
        print("i: %d, loss: %f"%(i,loss))
    if loss<0.1:
        break
plt.plot(loss_list)
plt.show()

plt.scatter(c_1[:,0],c_1[:,1])
plt.scatter(c_0[:,0],c_0[:,1],marker="+")
x=np.arange(10)  # 根据这个图 大概看出来 X 大概到10 左右
y=-(w[0]*x)/w[1]
plt.plot(x,y)
plt.show()

# 预测
# y_hat=w1*x1+w2*x2
y_hat=np.dot(w.T,test_data_x.T)
print(y_hat)
y_pred=np.array(y_hat<1,dtype=int)#.flatten()
print(y_pred)
test_acc=(np.sum(test_data_y==y_hat))/len(y_pred)
# !!!!!!!!!!! 准确率有问题
print(test_acc)





