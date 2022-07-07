# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline

class BPnn(torch.nn.Module):
    def __init__(self,
                 idim: int,
                 hidden_units1: int,
                 hidden_units2: int,
                 odim: int,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()

        self.hidden_layer1 = torch.nn.Linear(idim, hidden_units1)
        self.hidden_layer2 = torch.nn.Linear(hidden_units1,hidden_units2)
        self.out_layer = torch.nn.Linear(hidden_units2,odim)
        self.activation = activation

    def forward(self,x):
        hidden_x1_ = self.activation(self.hidden_layer1(x))
        hidden_x2_ = self.activation(self.hidden_layer2(hidden_x1_))
        y_ = self.out_layer(hidden_x2_)
        return y_

def picture_path(index):
    path1 = 'C:/Users/040/040/imag/'
    path2 = path1 + '0'
    if index < 10:
        img_path = path2 + str(index) + '.jpg'
    elif index in range(10, 13):
        img_path = path1 + str(index) + '.jpg'
    else:
        img_path = path1 + str(index) + '.bmp'
    return img_path


def get_img(index):
    path = picture_path(index)
    imag = cv2.imread(path, 0)
    result = cv2.resize(imag, (25, 25))
    res = np.resize(result, (1, 25 * 25))
    X1.append(res)

def div_data(test_n,X,Y):
    test_index = np.random.choice(len(X), test_n, replace=False)
    x_test = X[test_index]
    y_test = Y[test_index]

    train_index = np.arange(len(X))
    train_index = np.delete(train_index, test_index)
    x_train = X[train_index]
    y_train = Y[train_index]
    return test_index,x_test,x_train,y_test,y_train


def main():

    # 准备数据
    np.random.seed(100)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(1, 23):
        get_img(i)
    X = np.array(X1).squeeze()
    Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # 随机分割训练集和测试集
    test_index, x_test, x_train, y_test, y_train = div_data(12,X,Y)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)    # (10,625)
    x_test = std.transform(x_test)  # (12,625)

    x_train_ = torch.from_numpy(x_train)
    x_train_ = x_train_.float()
    x_test_ = torch.from_numpy(x_test).float()
    y_train_ = torch.from_numpy(y_train).long()
    y_test_ = torch.from_numpy(y_test).long()

    idim,h_unit1,h_unit2,odim,epoch = [625,20,20,2,200]
    model = BPnn(idim,h_unit1,h_unit2,odim)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.02,momentum=0.005)
    loss_func = torch.nn.CrossEntropyLoss()

    # train
    loss_ = []
    for step in range(epoch):
        optimizer.zero_grad()
        y_ = model(x_train_)
        loss = loss_func(y_, y_train_)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print("已训练{}步|loss：{}.".format(step,loss))
            a = loss.tolist()
            loss_.append(a)


    model.eval()
    print(model)

    y_pre = model(x_test_)
    y_pre = torch.log_softmax(y_pre,dim=0)
    y_pre_ = torch.max(y_pre, 1)[1]
    y_pre_ = y_pre_.numpy()
    accu = float((y_pre_ == y_test).astype(int).sum()) / float(y_test.size)*100
    print(y_pre_)
    print(y_test)
    print("accuracy: {}%".format(accu))
    cnt = 0
    for i in test_index:
        path = picture_path(i + 1)
        img = plt.imread(path)
        cnt = cnt + 1
        plt.subplot(2, 6, cnt)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        st = '图片ID：' + str(i) + '\n' + '所属种类' + str(Y[i]) + '\n' + '预测种类:' + str(y_pre_[cnt - 1])
        plt.title(st)
    plt.show()

    num_i = np.zeros(20)
    for i in range(20):
        num_i[i] = 10*i
    x_smooth = np.linspace(num_i.min(), num_i.max(), 200)
    y_smooth = make_interp_spline(num_i, loss_)(x_smooth)
    plt.plot(x_smooth, y_smooth)
    plt.xticks(num_i)
    plt.xlabel('迭代次数/次')
    plt.title('损失函数曲线图')
    plt.show()
    print("end")




    # 分类




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X1 = []
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
