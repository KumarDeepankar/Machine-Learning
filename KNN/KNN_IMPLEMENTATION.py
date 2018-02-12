import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
import operator

def ecludiandist(train_vector, test_vector):               # Calculating distance between vector
    distance = 0


    for i in range(len(train_vector) - 1):
        distance += pow((train_vector[i] - test_vector[i]), 2)
    return math.sqrt(distance)


def knnclassifier(trainset,testset):
    class_count = 0
    for i in range(len(testset)):
        distances = []

        for j in range(len(trainset)):

            ed=ecludiandist(trainset[j], testset[i])
            distances.append((trainset[j], ed))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(3):
            neighbors.append(distances[x][0])
        count =np.zeros(3,dtype=int)
        for t in range(len(neighbors)):
            neighbour_vote = neighbors[t]
            count[t]=neighbour_vote[9]
        predicted_class = np.argmax(np.bincount(count))
        test_class = testset[i,9]

        if predicted_class ==test_class:

            class_count += 1
    print("Accuracy of KNN Classifier is: " + str(class_count/len(testset)))  # Calculating accuracy


data = np.genfromtxt("breast-cancer-wisconsin.txt",dtype=int,delimiter=",")
train_row = int(len(data) * 80 / 100)
np.random.shuffle(data)
training, test = data[:train_row,:], data[train_row:,:]       #Data is splitted into 80 - 20 ratio
train_y = training[:,10]
train_x = training[:,[1,2,3,4,5,6,7,8,9,10]]                   #Constructing train feature vector

test_y = test[:,10]
test_x = test[:,[1,2,3,4,5,6,7,8,9,10]]                        #Constructing test feature vector

knnclassifier(train_x, test_x)
