import pandas as pd
import numpy as np
from numpy.linalg import inv


def categorical_var_to_column():                      # 1 Converting categorical variable to column binary represetation

    pd_df = pd.read_csv('lr.txt', sep=",", header = None)

    pd_df['F'] = 0
    pd_df['I'] = 0
    pd_df['M'] = 0

    pd_df['F'] = pd_df[0].apply(lambda x: 1 if x == 'F' else 0)
    pd_df['I'] = pd_df[0].apply(lambda x: 1 if x == 'I' else 0)
    pd_df['M'] = pd_df[0].apply(lambda x: 1 if x == 'M' else 0)

    return pd_df


def standardize_for_regression(pd_df):              # 2 Standardizing the independent variables

    row, col = pd_df.shape
    pd_df = pd_df.drop(labels=0, axis=1)
    pd_df = pd_df.drop(labels=1, axis=1)           # 3 Dropping two least significant attribute
    pd_df = pd_df.drop(labels=3, axis=1)
    Y = pd_df[8]
    X = pd_df.ix[:, pd_df.columns != 8]
    x_mean = X.mean()
    x_std = X.std()

    X = (X - x_mean) / (x_std)
    X.insert(0, 'b', value=1)
    X['Y'] = Y

    return X


def train_test(pd_df, partition_fract):             # 4 partitioning the data into train and test sets

    trains = pd_df.sample(frac=partition_fract)
    tests = pd_df.loc[~pd_df.index.isin(trains.index)]

    return trains, tests


def mylinridgereg(X,Y,ld):                        # 5 mylinridgereg(X,Y,ld) function implementation

    XT = np.matrix.transpose(np.array(X))
    X_row, Y_col = XT.shape

    idm = np.identity(X_row)
    lamda_i = ld * idm

    temp = np.dot(XT, np.array(X))

    temp1 = temp + lamda_i
    temp1_inv = inv(temp1)

    temp2 = np.dot(temp1_inv, XT)
    coef = np.dot(temp2, Y)
    print("Weights: "+ str(coef))
    return coef


def mylinridgeregeval(test_X, weights):            # 6 mylinridgeregeval function implementation

        k = [a * b for a, b in zip(test_X, weights)]
        return sum(k)


def meansquarederr(eval_list):                    # 7 meansquarederr function implementation

    sum_err = 0
    for i in eval_list:
        actual = i[0]
        predicted = i[1]
        err = actual - predicted
        sqr_err = err * err
        sum_err = sum_err + sqr_err
    mean_square_err = sum_err / len(eval_list)
    return mean_square_err



pd_df = categorical_var_to_column()
pd_df_s = standardize_for_regression(pd_df)

train, test = train_test(pd_df_s, 0.8)
df_row, df_col = train.shape

X_train = train.iloc[:,0:df_col - 1]
Y_train = train.iloc[:,df_col - 1]

X_test = test.iloc[:,0:df_col - 1]
Y_test = test.iloc[:,df_col - 1]

np_test_x = np.array(X_test)
np_test_y = np.array(Y_test)


def optimal_labmda(lamb):

    coefs = mylinridgereg(X_train,Y_train,lamb)

    eval_list=[]
    for i in range(0,len(np_test_y)):
        x_y_list = []
        prediction = mylinridgeregeval(np_test_x[i],coefs)
        x_y_list.append(np_test_y[i])
        x_y_list.append(prediction)
        eval_list.append(x_y_list)

        #print("Actual: " +str(x_y_list[0]) +" | Predicted: "+ str(x_y_list[1]))
        x_y_list = []

    mean_square_error = meansquarederr(eval_list)
    print("Mean Square Error: "+ str(mean_square_error))


labmda_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]   # 8 Trying out different lambda values
for i in labmda_list:
    print("Labmda value: " + str(i))
    optimal_labmda(i)





