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
import sys


file_path = sys.argv[1]                         #reading the input file


def replace_non_numeric(df):                    # Converting string to integer representation
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


def k_leaf_node_classifier():

    train_df = load_df.head(140)                                    # first 140 samples for training

    labels = train_df["KM_Overall_survival_censor"].values
    features = train_df[list(columns)].values

    clf = tree.DecisionTreeClassifier(criterion="entropy")         #scikit method
    clf = clf.fit(features, labels)

    test_df = load_df.tail(10)                                      # last 10 samples for testing

    test_labels = test_df["KM_Overall_survival_censor"].values
    test_features = test_df[list(columns)].values

    accuracy_c = clf.score(test_features, test_labels)
    print("Accuracy of the classifier is : " +str(accuracy_c))

    with open("decision_tree_3a.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,feature_names=train_df.columns.values)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("decision_tree_3a.png")

load_df = replace_non_numeric(pd.read_csv(file_path))


columns = ["Local_tumor_recurrence", "Gender","HPV/p16_status","Age_at_diagnosis","Race",
           "Tumor_side","Tumor_subsite","T_category","N_category","AJCC_Stage","Pathological_grade",
           "Smoking_status_at_diagnosis","Smoking_Pack-Years","Radiation_treatment_course_duration",
           "Total_prescribed_Radiation_treatment_dose","#_Radiation_treatment_fractions",
           "Induction_Chemotherapy","Concurrent_chemotherapy"]

load_df["Smoking_Pack-Years"] = load_df["Smoking_Pack-Years"].fillna(0)


k_leaf_node_classifier()


