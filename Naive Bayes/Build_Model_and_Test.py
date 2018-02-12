import numpy as np
from nltk.tokenize import WhitespaceTokenizer
from collections import OrderedDict
import copy
import pandas as pd
from sklearn.naive_bayes import BernoulliNB

train_txt = open('testdata')
Dictionary_txt = open('Dictionary')
result_txt = open('results.txt','w')
preprocessed = []
dic_list = OrderedDict()

for line1 in Dictionary_txt:
    dic_list[line1.strip()] = 0

temp_dic = OrderedDict()
count = 0
for line in train_txt:
    feature_vect = []
    temp_dic = copy.copy(dic_list)
    token = WhitespaceTokenizer().tokenize(line)
    for wrd in token:
       if wrd in temp_dic:
           #print(wrd)
           temp_dic[wrd] = 1
    for key in temp_dic:
        feature_vect.append(temp_dic[key])
    temp_dic.clear()
    str1 = ','.join(str(e) for e in feature_vect)

    preprocessed.append(feature_vect)

    #print(feature_vect)
    count += 1

Dictionary_txt.close()
train_txt.close()
x_test = np.matrix(preprocessed)


pd_df1 = pd.read_csv('testlabels', sep=" ", header = None)
y_test = np.array(pd_df1)

pd_df = pd.read_csv('processed_train.txt', sep=",", header = None)
row, col = pd_df.shape
x_train = pd_df.iloc[:,0:col  - 1]

y_train = pd_df.iloc[:,col - 1]
clf = BernoulliNB(alpha=1.2, binarize=0.0, class_prior=[1,2], fit_prior=True)
clf.fit(x_train, y_train)
print("Accuracy of the Naive Bayes classifier is " + str(clf.score(x_test,y_test)))
pr3 = clf.predict(x_test)
for i in pr3:
    result_txt.write(str(i))
    result_txt.write("\n")
result_txt.close()



