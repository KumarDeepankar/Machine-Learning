import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO
import os
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import sys
file_path = sys.argv[1]

def replace_non_numeric(df):
    df["Gender"] = df["Gender"].apply(lambda sex: 0 if sex == "Male" else 1)
    df["HPV/p16_status"] = df["HPV/p16_status"].apply(lambda hpv: 0 if hpv == "Positive" else 1)
    df["Race"] = df["Race"].apply(lambda race: 0 if race == "White" else 1 if race=="Asian" else 2 if race =="Hispanic" else 3)

    df["Tumor_side"] = df["Tumor_side"].apply(lambda tumor: 0 if tumor == "L" else 1 )
    df["Tumor_subsite"] = df["Tumor_subsite"].apply(lambda ts: 0 if ts == "BOT" else 1 if ts =="GPS" else 2 if ts =="Other"
                                                     else 3 if ts =="Pharyngeal_wall" else 4 if ts=="Soft_palate" else 5)

    df["N_category"] = df["N_category"].apply(lambda nc: 1 if nc == "1" else 2 if nc == "2" else 3 if nc =="3"
                                              else 4 if nc =="2a" else 5 if nc =="2b" else 6)

    df["AJCC_Stage"] = df["AJCC_Stage"].apply(lambda a_s: 0 if a_s == "I" else 1 if a_s =="II" else 2 if a_s =="III"
                                              else 3)

    df["Pathological_grade"] = df["Pathological_grade"].apply(lambda p_g: 0 if p_g == "I" else 1 if p_g == "II"
                                             else 2 if p_g =="III" else 3 if p_g =="IV" else 4 if p_g =="II-III" else 5)

    df["Smoking_status_at_diagnosis"] = df["Smoking_status_at_diagnosis"].apply(lambda s_s: 0 if s_s == "Current"
                                             else 1 if s_s=="Former" else 2)

    df["Induction_Chemotherapy"] = df["Induction_Chemotherapy"].apply(lambda i_c: 0 if i_c == "Y" else 1)
    df["Concurrent_chemotherapy"] = df["Concurrent_chemotherapy"].apply(lambda c_c: 0 if c_c == "Y" else 1)


    return df



accuracy_c = 0
def k_leaf_node_classifier(k):

    train_df = load_df.head(140)

    labels = train_df["KM_Overall_survival_censor"].values
    features = train_df[list(columns)].values

    clf = tree.DecisionTreeClassifier(criterion="entropy", max_features=10,max_leaf_nodes = k)
    clf = clf.fit(features, labels)

    test_df = load_df.tail(10)

    test_labels = test_df["KM_Overall_survival_censor"].values
    test_features = test_df[list(columns)].values

    predictions = clf.predict(test_features)

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    num_leaf = np.bincount(is_leaves)
    #print(num_leaf[1])
    #print(clf.score(test_features, test_labels))
    global accuracy_c
    accuracy_c = clf.score(test_features, test_labels)

load_df = replace_non_numeric(pd.read_csv(file_path))

columns = ["Local_tumor_recurrence", "Gender","HPV/p16_status","Age_at_diagnosis","Race",
           "Tumor_side","Tumor_subsite","T_category","N_category","AJCC_Stage","Pathological_grade",
           "Smoking_status_at_diagnosis","Smoking_Pack-Years","Radiation_treatment_course_duration",
           "Total_prescribed_Radiation_treatment_dose","#_Radiation_treatment_fractions",
           "Induction_Chemotherapy","Concurrent_chemotherapy"]


load_df["Smoking_Pack-Years"] = load_df["Smoking_Pack-Years"].fillna(0)


leaf_node_array = np.empty((0,2))

for i in range(10):

    print(5 +3*i)
    k = 5 + 3*i
    leaf_node_array = np.append(leaf_node_array,np.array([[k,accuracy_c]]), axis= 0)
    k_leaf_node_classifier(5 +3*i)

print(leaf_node_array)
plt.plot(leaf_node_array[:,0],leaf_node_array[:,1])
plt.ylabel('Classification accuracy')
plt.xlabel('Number of leaf node')
plt.show()
