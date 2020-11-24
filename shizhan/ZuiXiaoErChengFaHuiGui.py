''''#最小二乘法求一元线性回归
#y=4x+2
#构造数据集'''
import numpy as np
import  matplotlib.pyplot as plt
np.random.seed(1)
x=np.random.normal(size=(100,1),scale=1)
#print(type(x))
'''numpy中 numpy.random.normal(loc=0.0, scale=1.0, size=None)  
参数的意义为：
　　loc:float     概率分布的均值，对应着整个分布的中心center
　　scale:float   概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
　　size:int or tuple of ints    输出的shape，默认为None，只输出一个值
　　我们更经常会用到np.random.randn(size)所谓标准正太分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)
'''
y=4*x[:,0]+2
plt.scatter(x,y)
plt.show()
all_data=np.concatenate((x,y.reshape(100,1)),axis=1)
#print(all_data.shape)
#分训练集 测试机
np.random.shuffle(all_data)
train_data = all_data[:70,:]
test_data  = all_data[70:100,:]
#构造模型  y=W x+b
w=np.random.rand()
# 先把w转换成列表，再把B转换成数组,,真费劲
listw=[]
listw.append(w)
#print(listw)
v=np.array(listw)

b=np.random.rand()
# 先把B转换成列表，再把B转换成数组
listc=[]
listc.append(b)
#print(listc)
c=np.array(listc)
#定义损失函数   平方损失
#超参数
lr=0.001
#构造权重的增广向量  w b  放在一起
w_hat=np.concatenate((v.reshape(1,1),c.reshape(1,1)))
w_hat=w_hat.reshape(2,1)
print("w_hat的样子",w_hat.shape)

x=train_data[:,:-1]
print(x.shape)
x=x.reshape(1,70)#一行70列
#print(x)
#print(x.ndim)#   2维的
y=train_data[:,-1]
#x.reshape()

#增广的特征向量
x_hat=np.concatenate((x,np.ones((1,70))))
print("x_hat的样子",x_hat.shape)

# 算法实现
num=1
w_hat_list=[]
b_hat_list=[]
loss_list=[]
while True:
    #更新参数
    w_hat=w_hat+lr*np.dot(x_hat,(y.reshape(70,1)-np.dot(x_hat.T,w_hat)))
    #计算经验错误
    loss=np.sum((y.reshape(70,1)-np.dot(x_hat.T,w_hat))**2)/2
   #loss=np.sum(0.5 * (theta * x - y) ** 2)  # 损失之和
    w_hat_list.append(w_hat[0])
    b_hat_list.append(w_hat[1])
    loss_list.append(loss)
    num=num+1
    print("num: %d ,loss：  %f"%(num,loss))
    if loss<1 or num>1000:
        break
print(loss,w_hat_list[-1],b_hat_list[-1])
plt.plot(w_hat_list)
plt.show()
plt.plot(b_hat_list)
plt.show()

