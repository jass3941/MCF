# -*- coding: utf-8 -*-

"""
# df.shape[0] # number of row -> cf item
# df.shape[1] # number of time horizon of cf

# min-max normailzation을 사용하여 cf 업데이트
# drop empty items if necessary
"""
import pandas as pd
import numpy as np


# def min_max_scale_daham(df1):
#    df2 = pd.DataFrame(columns = df1.columns)
#    df2.iloc[:,0] = df1.iloc[:,0]
#    for j in range(1 , df1.shape[1]) : # number of cols --> (cf item)
#        minimum = df1.iloc[:,j].min()
#        maximum = df1.iloc[:,j].max()
#        for i in range(df1.shape[0]) : # number of rows --> (time horizon of cf)
#            df2.iloc[i,j] = (df1.iloc[i,j]- minimum) / (maximum - minimum)
#    return df2
# df_norm = min_max_scale_daham(cf)

def min_max_sclae(df1):
    import pandas as pd
    df2 = pd.DataFrame(columns=df1.columns)
    df2.iloc[:, 0] = df1.iloc[:, 0]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df1.iloc[:, 1:])
    temp = pd.DataFrame(scaler.transform(df1.iloc[:, 1:]))

    for i in range(0, df1.shape[1]):
        df2.iloc[:, i] = temp.iloc[:, i - 1]
    return df2


# df_norm = min_max_sclae(cf)


def diff_normalization(df1):
    df2 = pd.DataFrame(columns=df1.columns)
    df2.iloc[:, 0] = df1.iloc[:, 0]

    for j in range(0, df1.shape[1]):  # number of cols --> (cf item)
        for i in range(1, df1.shape[0]):  # number of rows --> (time horizon of cf)
            df2.iloc[i, j] = df1.iloc[i, j] - df1.iloc[i - 1, j]
    return df2


# df_diff = diff_normalization(cf)

def cosine(u, v, w=None):
    # umu = np.average(u, weights=w)
    # vmu = np.average(v, weights=w)
    # u = u - umu
    # v = v - vmu
    # uv = np.average(u * v, weights=w)
    # uu = np.average(np.square(u), weights=w)
    # vv = np.average(np.square(v), weights=w)
    # dist = 1.0 - uv / np.sqrt(uu * vv)

    uv = np.average(np.dot(u, v), weights=w)
    uu = np.average(np.dot(u, u), weights=w)
    vv = np.average(np.dot(v, v), weights=w)

    dist = uv / np.sqrt(uu * vv)
    return dist


def cos_similarity(df1, df2):
    # from scipy.spatial import distance
    similarity_vector = []
    for i in range(0, df1.shape[1]):
        test1 = df1.iloc[:, i].to_numpy()
        test2 = df2.iloc[:, i].to_numpy()
        similarity_vector.append(cosine(test1, test2))
    return similarity_vector


def size_similarity(df1, df2):
    similarity_vector = []
    for i in range(0, df1.shape[1]):
        test1 = sum(df1.iloc[:, i])
        test2 = sum(df2.iloc[:, i])
        if test2 == 0:
            temp = 0
        else:
            temp = test1 / test2
        similarity_vector.append(temp)
    return similarity_vector
