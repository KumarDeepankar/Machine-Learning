import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


train_plot_list_x = []
train_plot_list_y = []
test_plot_list_x = []
test_plot_list_y = []

def categorical_var_to_column():

    pd_df = pd.read_csv('lr.txt', sep=",", header = None)

    pd_df['F'] = 0
    pd_df['I'] = 0
    pd_df['M'] = 0

    pd_df['F'] = pd_df[0].apply(lambda x: 1 if x == 'F' else 0)
    pd_df['I'] = pd_df[0].apply(lambda x: 1 if x == 'I' else 0)
    pd_df['M'] = pd_df[0].apply(lambda x: 1 if x == 'M' else 0)

    return pd_df


def standardize_for_regression(pd_df):

    row, col = pd_df.shape
    pd_df = pd_df.drop(labels=0, axis=1)
    pd_df = pd_df.drop(labels=1, axis=1)
    pd_df = pd_df.drop(labels=3, axis=1)
    Y = pd_df[8]
    X = pd_df.ix[:, pd_df.columns != 8]
    x_mean = X.mean()
    x_std = X.std()

    X = (X - x_mean) / (x_std)
    X.insert(0, 'b', value=1)
    X['Y'] = Y

    return X


def train_test(pd_df, partition_fract):

    trains = pd_df.sample(frac=partition_fract)
    tests = pd_df.loc[~pd_df.index.isin(trains.index)]

    return trains, tests


def mylinridgereg(X,Y,ld):

    XT = np.matrix.transpose(np.array(X))
    X_row, Y_col = XT.shape

    idm = np.identity(X_row)
    lamda_i = ld * idm

    temp = np.dot(XT, np.array(X))

    temp1 = temp + lamda_i
    temp1_inv = inv(temp1)

    temp2 = np.dot(temp1_inv, XT)
    coef = np.dot(temp2, Y)
    #print("Weights: "+ str(coef))
    return coef

def mylinridgeregeval(test_X, weights):

        k = [a * b for a, b in zip(test_X, weights)]
        return sum(k)

def meansquarederr(eval_list):

    sum_err = 0
    for i in eval_list:
        actual = i[0]
        predicted = i[1]
        err = actual - predicted
        sqr_err = err * err
        sum_err = sum_err + sqr_err
    mean_square_err = sum_err / len(eval_list)
    return mean_square_err

def optimal_labmda(X_train,Y_train,np_test_x, np_test_y, lamb):

    coefs = mylinridgereg(X_train,Y_train,lamb)

    eval_list_test=[]
    for i in range(0,len(np_test_y)):
        x_y_list_test = []
        prediction = mylinridgeregeval(np_test_x[i],coefs)
        x_y_list_test.append(np_test_y[i])
        x_y_list_test.append(prediction)
        eval_list_test.append(x_y_list_test)
        test_plot_list_x.append(prediction)
        test_plot_list_y.append(np_test_y[i])

        #print("Actual: " +str(x_y_list[0]) +" | Predicted: "+ str(x_y_list[1]))
        x_y_list_test = []

    mean_square_error = meansquarederr(eval_list_test)
    print("Mean Square Error for test data: "+ str(mean_square_error))


    eval_list_train = []
    np_Y_train = np.array(Y_train)
    np_X_train = np.array(X_train)
    for i in range(0, len(np_Y_train)):
        x_y_list_train = []
        prediction = mylinridgeregeval(np_X_train[i], coefs)
        x_y_list_train.append(np_Y_train[i])
        x_y_list_train.append(prediction)
        eval_list_train.append(x_y_list_train)
        train_plot_list_x.append(prediction)
        train_plot_list_y.append(np_Y_train[i])

        # print("Actual: " +str(x_y_list[0]) +" | Predicted: "+ str(x_y_list[1]))
        x_y_list_train = []

    mean_square_error = meansquarederr(eval_list_train)
    print("Mean Square Error for train data: " + str(mean_square_error))




def plot_chart(split_fraction):

    pd_df = categorical_var_to_column()
    pd_df_s = standardize_for_regression(pd_df)

    train, test = train_test(pd_df_s, split_fraction)
    df_row, df_col = train.shape

    X_train = train.iloc[:,0:df_col - 1]
    Y_train = train.iloc[:,df_col - 1]

    X_test = test.iloc[:,0:df_col - 1]
    Y_test = test.iloc[:,df_col - 1]

    np_test_x = np.array(X_test)
    np_test_y = np.array(Y_test)
    labmda_list = [0.02]
    for i in labmda_list:
        print("Labmda value: " + str(i))
        optimal_labmda(X_train,Y_train,np_test_x, np_test_y, i)


plot_chart(0.8)

fig1 = plt.figure(1)
fig1.suptitle('Partition Fraction: 0.8, Lambda: 0.02', fontsize=20)
plt.plot(test_plot_list_x, test_plot_list_y)
plt.xlabel("Test Predicted Values")
plt.ylabel("Test Actual Values")
plt.show()

fig2 = plt.figure(2)
fig2.suptitle('Partition Fraction: 0.8, Lambda: 0.02', fontsize=20)
plt.plot(train_plot_list_x, train_plot_list_y)
plt.xlabel("Training Predicted Values")
plt.ylabel("Training Actual Values")
plt.show()