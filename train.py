"""
PHuong huong moi: chinh sua dataTest == dataTrain, ham danh gia score
# data_test, label_test = fromdtlbToDtLb(data_test)
"""

import numpy as np
import pandas as pd
import os
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

def fromdtlbToDtLb(train):
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
# dataNp = pd.read_csv("dataUniqueL2.csv", index_col=0).to_numpy()
# # dataNp = np.random.randint(0, 100, size=(10000, 27))
# n_end = 100
# test = dataNp[-n_end:]
# train = dataNp[:-n_end]
# x_train, y_train = fromdtlbToDtLb(train) 
# # model = ComplementNB()

# def runTest(model):
#     model.fit(x_train, y_train)
#     l = len(test)-1
#     isTrue = 0
#     money = 0
#     hs = []
#     for i in range(l):
#         prd = model.predict([test[i]])
#         results = test[i+1]
#         counter =  np.count_nonzero(results == prd)
#         # print("\n",prd)
#         # print(results)
#         # print(counter)
#         if counter>0:
#             isTrue+=1
#             money += (counter*80 -23) 
#         else:
#             money -= 23
#             pass
#         # print(money)
#         hs.append(money)
#     # print(isTrue/l)
#     # print(money)
#     return np.array(hs)

# hs1 = runTest(GaussianNB())
# hs2 = runTest(MultinomialNB())
# hs3 = runTest(BernoulliNB())
# hs4 = runTest(ComplementNB())
# hsmean = (hs1 + hs2 + hs3 + hs4) / 4
# x = np.arange(len(hs1))

# # Vẽ biểu đồ
# plt.figure(figsize=(8, 6))
# plt.plot(x, hs1, linestyle='-', color='red', label='GaussianNB')
# plt.plot(x, hs2, linestyle='-', color='green', label='MultinomialNB')
# plt.plot(x, hs3, linestyle='-', color='blue', label='BernoulliNB')
# plt.plot(x, hs4, linestyle='-', color='yellow', label='ComplementNB')
# plt.plot(x, hsmean, linestyle='-', color='black', label='Mean')

# # Thêm tiêu đề và nhãn trục
# plt.title('Biểu Đồ Các Phần Tử Của Mảng')
# plt.xlabel('Chỉ số')
# plt.ylabel('Giá trị')

# # Thêm chú thích
# plt.legend()

# # Hiển thị biểu đồ
# plt.grid(True)
# plt.show()


# dataTrain
# dataTest
# dataPredict


def Train(model_new, filename, data_train, label_train, data_test, label_test):
    model_old = None 
    if os.path.exists(filename):
        model_old = joblib.load(filename)
    model_new.fit(data_train, label_train)
    model_new.maxscore = model_new.score(data_test, label_test)
    # model_new.money = 0
    # model_new.history = []
    # l = len(dtLbTest)-1
    # for i in range(l):
    #     prd = model_new.predict([dtLbTest[i]])
    #     rss = dtLbTest[i+1]
    #     counter = np.count_nonzero(rss == prd)
    #     # print(prd)
    #     # print(rss)
    #     # print(counter)
    #     if counter>0:
    #         model_new.money += (counter*80 -23)
    #     else:
    #         model_new.money -= 23
    #     model_new.history.append(model_new.money)
    #     # print(model_new.history)
    if (not model_old) or model_new.maxscore > model_old.maxscore:
        print(f"newmodel = {model_new.maxscore}")
        joblib.dump(model_new, filename)

dataNp = pd.read_csv("dataUniqueL2.csv", index_col=0).to_numpy()
n_end = 100
dtLbTest = dataNp[:-n_end]
data_test, label_test = fromdtlbToDtLb(dtLbTest)
dtLbPredict = dataNp[-n_end:]

def runTrain(model, fileName, data_test, label_test):
    num_episodes=100
    for i in range(num_episodes):
        print(f"\n\nEpisodes {i}")
        data_random = np.random.randint(0, 100, size=(len(data_test), 27))
        data_train, label_train = fromdtlbToDtLb(data_random)
        Train(model, fileName, data_train, label_train, data_test, label_test)

def runPredict(model, dtLbPredict):
    # model = joblib.load('GaussianNB.pkl')
    # plt.plot(np.arange(len(model.history)), model.history, linestyle='-', color='green', label='test_GaussianNB')
    money = 0
    history = []
    l = len(dtLbPredict)-1
    for i in range(l):
        prd = model.predict([dtLbPredict[i]])
        rss = dtLbPredict[i+1]
        counter = np.count_nonzero(rss == prd)
        if counter>0:
            money += (counter*80 -23)
        else:
            money -= 23
        history.append(money)

    x = np.arange(len(history))
    plt.plot(x, history, linestyle='-', color='red', label='predict_GaussianNB')
    plt.show()

runTrain(GaussianNB(), 'GaussianNB.pkl', data_test, label_test)
runPredict(joblib.load('GaussianNB.pkl'), dtLbPredict)