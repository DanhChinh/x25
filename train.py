import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
# from sklearn.metrics import accuracy_score
import joblib
# joblib.dump(model, 'model.pkl')
# model = joblib.load('model.pkl')
import matplotlib.pyplot as plt

def convertTrain(train):
    data = []
    label = []
    l = len(train)-1
    for i in range(l):
        data.append(train[i])
        # label.append(train[i+1][0])
        results = train[i+1]
        counts = np.bincount(results)
        most_frequent = np.argmax(counts)
        label.append(most_frequent)
    return np.array(data), np.array(label)

dataNp = pd.read_csv("dataUniqueL2.csv", index_col=0).to_numpy()
# dataNp = np.random.randint(0, 100, size=(10000, 27))
n_end = 100
test = dataNp[-n_end:]
train = dataNp[:-n_end]
x_train, y_train = convertTrain(train) 
# model = ComplementNB()

def runTest(model):
    model.fit(x_train, y_train)
    l = len(test)-1
    isTrue = 0
    money = 0
    hs = []
    for i in range(l):
        prd = model.predict([test[i]])
        results = test[i+1]
        counter =  np.count_nonzero(results == prd)
        # print("\n",prd)
        # print(results)
        # print(counter)
        if counter>0:
            isTrue+=1
            money += (counter*80 -23) 
        else:
            money -= 23
            pass
        # print(money)
        hs.append(money)
    # print(isTrue/l)
    # print(money)
    return np.array(hs)

hs1 = runTest(GaussianNB())
hs2 = runTest(MultinomialNB())
hs3 = runTest(BernoulliNB())
hs4 = runTest(ComplementNB())
hsmean = (hs1 + hs2 + hs3 + hs4) / 4
x = np.arange(len(hs1))

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(x, hs1, linestyle='-', color='red', label='GaussianNB')
plt.plot(x, hs2, linestyle='-', color='green', label='MultinomialNB')
plt.plot(x, hs3, linestyle='-', color='blue', label='BernoulliNB')
plt.plot(x, hs4, linestyle='-', color='yellow', label='ComplementNB')
plt.plot(x, hsmean, linestyle='-', color='black', label='Mean')

# Thêm tiêu đề và nhãn trục
plt.title('Biểu Đồ Các Phần Tử Của Mảng')
plt.xlabel('Chỉ số')
plt.ylabel('Giá trị')

# Thêm chú thích
plt.legend()

# Hiển thị biểu đồ
plt.grid(True)
plt.show()