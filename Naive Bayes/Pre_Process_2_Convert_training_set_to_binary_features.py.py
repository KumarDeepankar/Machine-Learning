import numpy as np
from nltk.tokenize import WhitespaceTokenizer
from collections import OrderedDict
import copy

preprocessed = open('preprocessed.txt','w')
train_txt = open('traindata')
Dictionary_txt = open('Dictionary')
tarin_label = open('trainlabels')
target_vect = []
for line0 in tarin_label:
    target_vect.append(line0.strip())

dic_list = OrderedDict()

for line1 in Dictionary_txt:
    dic_list[line1.strip()] = 0
    preprocessed.write(line1.strip() + ",")
preprocessed.write("\n")
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
    feature_vect.append(target_vect[count])
    print(feature_vect)
    str1 = ','.join(str(e) for e in feature_vect)
    preprocessed.write(str1)
    preprocessed.write("\n")
    count += 1

print(len(target_vect))
print(count)
Dictionary_txt.close()
train_txt.close()
tarin_label.close()
preprocessed.close()