import numpy as np
import pandas as pd

class KNN:
    def __init__(self, k):

        self.k = k
        pass

    def fit(self, x_train, y_train):

        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def get_k_neighbors(self, x_input):

        distance = []

        for idx, i in enumerate(self.x_train):
            dis = (np.sum((np.array(i) - np.array(x_input)) ** 2) )** 0.5

            distance.append([dis, self.y_train[idx]])

        distance.sort()

        arr = np.array(distance[:self.k])[:, -1]
        return arr

    def predict(self, x_input):

        self.pre_x = np.array(x_input)

        Y_pre = []

        for x in self.pre_x:

            k_nearst_neighbors = self.get_k_neighbors(x)
            a = pd.Series(k_nearst_neighbors).value_counts()

            y_pre = a.idxmax()

            Y_pre.append(y_pre)

        return Y_pre

    def score(self, x_input, y_input):

        h_hat = self.predict(x_input)

        acc = np.mean(h_hat == y_input)

        return acc


if __name__ == '__main__':
    T = np.array([
        [3, 104, -1],
        [2, 100, -1],
        [1, 81, -1],
        [101, 10, 1],
        [99, 5, 1],
        [98, 2, 1]])
    X_train = T[:, :-1]
    Y_train = T[:, -1]
    x_test = [[18, 90], [50, 50]]
    knn = KNN(k=5)
    knn.fit(x_train=X_train, y_train=Y_train)
    print(knn.predict(X_train))
    print(knn.predict(x_test))
    print(knn.score(x_input=X_train, y_input=Y_train))
    # knn.fetch_k_neighbors(x_test[0])
    print('预测结果：{}'.format(knn.predict(x_test)))
    print('-----------下面测试一下鸢尾花数据-----------')
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, Y = load_iris(return_X_y=True)
    print(X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print(x_train.shape, y_train.shape)
    knn01 = KNN(k=1)
    knn01.fit(x_train, y_train)
    print("训练集准确率{}".format(knn01.score(x_train, y_train)))
    print("测试集准确率{}".format(knn01.score(x_test, y_test)))
