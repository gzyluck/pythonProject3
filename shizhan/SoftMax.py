
#SoftMax实战
# 特征 2维，分类有四个类别，表示为：X（X1  X2）   Y(0/1/2/3)
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
num=100

x_1=np.random.normal(-3,1,size=(num))#  随机数，在3附近产生，产生规律为高斯分布
x_2=np.random.normal(-3,1,size=(num))
y=np.zeros(num)
c_0=np.array([x_1,x_2,y]).T
#print(c_0)

x_1=np.random.normal(3,1,size=(num))#  随机数，产生，产生规律为高斯分布
x_2=np.random.normal(-3,1,size=(num))
y=np.ones(num)
c_1=np.array([x_1,x_2,y]).T

x_1=np.random.normal(-3,1,size=(num))#  随机数，产生，产生规律为高斯分布
x_2=np.random.normal(3,1,size=(num))
y=np.ones(num)*2
c_2=np.array([x_1,x_2,y]).T

x_1=np.random.normal(3,1,size=(num))#  随机数，产生，产生规律为高斯分布
x_2=np.random.normal(3,1,size=(num))
y=np.ones(num)*3
c_3=np.array([x_1,x_2,y]).T

plt.scatter(c_0[:,0],c_0[:,1],marker="+")
plt.scatter(c_1[:,0],c_1[:,1],marker="*")
plt.scatter(c_2[:,0],c_2[:,1],marker=".")
plt.scatter(c_3[:,0],c_3[:,1])
plt.show()

all_data=np.concatenate((c_0,c_1,c_2,c_3))
print(all_data.shape)

np.random.shuffle(all_data)

train_data_x=all_data[:300,:2]
train_data_y=all_data[:300,-1].reshape(300,1)
test_data_x=all_data[300:,:2]
test_data_y=all_data[300:,-1].reshape(100,1)
print(test_data_x.shape,test_data_y.shape,train_data_x.shape,train_data_y.shape)

# y =w1 * x1+w2*x2 +b
# 0= w1*x +w2 * y +b
# Y=-（w1 *x +b）/w2
w=np.random.rand(4,2)
print(w.shape)

bias =np.random.rand(1,4)
print(bias.shape)

plt.scatter(c_0[:,0],c_0[:,1],marker="+")
plt.scatter(c_1[:,0],c_1[:,1],marker="*")
plt.scatter(c_2[:,0],c_2[:,1],marker=".")
plt.scatter(c_3[:,0],c_3[:,1])
x=np.arange(-5,5)
y1= -(w[0,0] * x + bias[0,0])/ w[0,1]
plt.plot(x,y1,'b')
y2= -(w[1,0] * x + bias[0,1])/ w[1,1]
plt.plot(x,y2,'y')
y3= -(w[2,0] * x + bias[0,2])/ w[2,1]
plt.plot(x,y3,'r')
y4= -(w[3,0] * x + bias[0,3])/ w[3,1]
plt.plot(x,y4,'g')
plt.show()


# softmax(x)=e^x/sum(e^x)
def softmax(z):
    exp=np.exp(z)
    sum_exp=np.sum( np.exp(z),axis=1,keepdims=True ) # 按照axis=1这个维度的权重加起来。
    return exp/sum_exp

'''
b= np.array(  [1,2,3,4,5,6] ).reshape(2,3)
print(softmax(b))

'''

# one_hot 编码
# temp 1,2,3,4 表示3  用向量  【0，0，1，0】
def one_hot(temp):
    one_hot=np.zeros( (len(temp),len(np.unique(temp))) )#例如 train_data_y  是 300行4列的元组
    one_hot[np.arange(len(temp)), temp.astype(np.int).T ]=1 #astype（） 改变np.array中所有数据元素的数据类型
    return one_hot
print(one_hot(train_data_y))
'''
    #np.arange(len(temp))        1行 300列
    #temp.astype(np.int).T       1行 300列
a=(len(train_data_y),len(np.unique(train_data_y)))
b=np.zeros( (2,2) )
print(b)
#print(train_data_y.shape) #300行 1列
#print(len(train_data_y))#len(train_data_y) 300行
print(len(np.unique(train_data_y) )) #4 个类型
#print(  train_data_y.astype(np.int)) # 0 1 2 3的数组
#print(np.arange(len(train_data_y)))
#print(one_hot(train_data_y))
'''
#  计算y_hat
def compute_y_hat(w,x,b):
    return np.dot(x,w.T)+b


#计算交叉熵
def cross_entroy(y,y_hat):
    loss= -(1/len(y))*np.sum(y*np.log(y_hat))
    return loss
# w= w- lr *grad
lr=0.01
all_loss=[]
for i in range(1000):
    #计算 loss
    x=train_data_x
    y=one_hot(train_data_y)
    y_hat=softmax(compute_y_hat(w,x,bias))
    loss=cross_entroy(y,y_hat)
    all_loss.append(loss)
    # 梯度更新
    grad_w = ((1 / len(x))) * np.dot(x.T, (y_hat - y))
    grad_b = ((1 / len(x))) * np.sum( (y_hat - y))
    #更新参数
    w= w -lr * grad_w.T
    bias= bias- lr * grad_b.T
    #输出
    if i%300==1:
        print("i : %d   , loss :  %f"% (i , loss))
plt.plot(all_loss)
plt.show()


plt.scatter(c_0[:,0],c_0[:,1],marker="+")
plt.scatter(c_1[:,0],c_1[:,1],marker="*")
plt.scatter(c_2[:,0],c_2[:,1],marker=".")
plt.scatter(c_3[:,0],c_3[:,1])
x=np.arange(-5,5)
y1= -(w[0,0] * x + bias[0,0])/ w[0,1]
plt.plot(x,y1,'b')
y2= -(w[1,0] * x + bias[0,1])/ w[1,1]
plt.plot(x,y2,'y')
y3= -(w[2,0] * x + bias[0,2])/ w[2,1]
plt.plot(x,y3,'r')
y4= -(w[3,0] * x + bias[0,3])/ w[3,1]
plt.plot(x,y4,'g')
plt.show()




#  测试
def predict(x):
    y_hat=softmax(compute_y_hat(w,x,bias))
    return np.argmax(y_hat,axis=1)[:,np.newaxis]
# 输出准确度
print(np.sum(predict(test_data_x)==test_data_y)/len(test_data_y))

















