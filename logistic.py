from sklearn import linear_model
import numpy as np
from scipy.stats.distributions import chi2
import pandas as pd
from sklearn.datasets  import make_classification
import os
import matplotlib.pyplot as plt

global dir_w
dir_w = os.getcwd()

# def make_data():
#     X, y = make_classification(n_features=10, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=4)
#     # df = pd.DataFrame()
#     # for i in range(10):
#     #     df["X%s" % str(i + 1)] = X[:, i]
#     # df.to_csv('\\classification_1.csv', header=False, index=False)
#     return X,y


## 로그 가능도 계산 MODEL1
def compute_likelihood(place):
    data = pd.read_csv(place)
    y = data[['F11']]
    X = data[data.columns.difference(['F11'])]
    global d1_num
    d1_num = len(X.columns)
    X = np.array(X)
    y = np.array(y)
    logit = linear_model.LogisticRegression()
    logit.fit(X, y)
    y = y.reshape(1, 100)
    z = np.multiply(X, logit.coef_)
    ll = np.sum(np.dot(y, z) - np.log(1 + np.exp(z)))
    #print('data1 모델의 가능도 : ', ll)
    return ll

## 로그 가능도 계산 MODEL2
def compute_likelihood_2(place):
    data = pd.read_csv(place)
    y = data[['F11']]
    X = data[data.columns.difference(['F11'])]
    global d2_num
    d2_num = len(X.columns)
    X = np.array(X)
    y = np.array(y)
    logit = linear_model.LogisticRegression()
    logit.fit(X, y)
    y = y.reshape(1, 100)
    z = np.multiply(X, logit.coef_)
    ll = np.sum(np.dot(y, z) - np.log(1 + np.exp(z)))
    #print('data1 모델의 가능도 : ', ll)
    return ll



## 로그 가능도비 계산
def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))



## 가능도검정 포화 모형(second) / 축소모형(first)
def compare_model(first,second):
    LR = likelihood_ratio(first, second) ## min,max
    # 자유도계산
    df = abs(d1_num - d2_num)
    if df == 0 :
       df = 1;
    else:
        df == df;
    ## H0 모형이 적합함
    ## 정규분포함을 전제
    p = chi2.sf(LR, df)
    return p


## 최대가능도
def maxlikehood():
    result = []
    index_num = []
    count = 0
    name = []
    data = pd.read_csv('C:\\toi\\Likehood\\classification.csv',header=None)
    y = data.iloc[:,10]
    X = data.iloc[:,0:11]
    global d2_num
    d2_num = len(X.columns)
    X = np.array(X)
    y = np.array(y)

    for i in range(0, 11):
        for j in range(0, 11):
            if j > i:
                count = count + 1
                # 로지스틱 회귀 모형 적합

                logit = linear_model.LogisticRegression()
                logit.fit(X[:, i:j], y)
                # 로그 가능도 계산
                #y = y.reshape(1, 100)
                z = np.multiply(X[:, i:j], logit.coef_)
                LR = np.sum(np.dot(y, z) - np.log(1 + np.exp(z)))
                result.append(LR)
                index_num.append([i, j])
                name_num = "{0} 번째 모델".format(count)
                name.append([name_num, LR, [i,j]])
            # elif j==i and i > 0 and j > 0 :
            #     count = count + 1
            #     # 로지스틱 회귀 모형 적합
            #     X_1 = X[:, j]
            #     X_1 = X_1.reshape(-1, 1)
            #     logit = linear_model.LogisticRegression()
            #     logit.fit(X_1, y)
            #     # 로그 가능도 계산
            #     z = np.multiply(X_1, logit.coef_)
            #     LR = np.sum(np.dot(y, z) - np.log(1 + np.exp(z)))
            #     result.append(LR)
            #     index_num.append([i, j])
            #     name_num = "{0} 번째 모형".format(count)
            #     name.append([name_num, LR, [i, j]])

            else:
                pass


    name_dp = pd.DataFrame(name)
    name_dp.columns = ['Model' ,'Likehood' , 'Feature']
    # for i in range(10):
    #     name_dp["X%s" % str(i + 1)] = X[:, i]
    global pl
    pl = 'C:\\toi\\Likehood\\data_result.csv'
    name_dp.to_csv(pl, encoding='euc-kr',header= True,index=False)

    value = max(result)
    plot_max()

    return value

def where_file():
    return pl

def plot_max():
    data = pd.read_csv('C:\\toi\\Likehood\\classification.csv', header=None)
    y = data.iloc[:, 10]
    x = data.iloc[:, 4:6]

    x = np.array(x)
    y = np.array(y)
    logit = linear_model.LogisticRegression()

    logit.fit(x, y)
    y = y.reshape(1, 100)
    z = np.multiply(x, logit.coef_)

    xes = np.arange(0, 100, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(xes, x)
    ax1.set_ylabel('X Value')

    ax2.plot(xes, z)
    ax2.set_ylabel('Z Value')
    ax2.yaxis.set_label_coords(-0.12, 0.5)

    fig.savefig('C:\\toi\\Likehood\\model.png')