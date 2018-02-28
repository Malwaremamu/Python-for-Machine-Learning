
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from collections import Counter
knn_test = pd.read_csv("MNIST_test.csv")
knn_train = pd.read_csv("MNIST_training.csv")


# In[2]:


label_training = np.array(knn_train.iloc[:,0]) #Label column for Training data set
training_set = np.array(knn_train.iloc[:,1:]) #Training data excluding labels
label_test = np.array(knn_test.iloc[:,0]) #label clumn for test data set
test_set = np.array(knn_test.iloc[:,1:]) #Test data excluding labels


# In[3]:


def euclidean_distance(a, b): #To calculate the Euclidean Distance
    distance = 0
    for i in range(len(a)):
        distance += pow((a[i] - b[i]),2) #(a-b)*2
    distance = math.sqrt(distance) #sqrt[(a-b)*2]
    return distance


# In[19]:


dist = []
label_predicted = []
kAccuracy = []
k = [3, 5, 7, 9 , 11] #K values for prediction
for j in range(len(k)):
    label_predicted = []
    for row1 in range(len(test_set)): 
        dist = []
        for row2 in range(len(training_set)):
            d = euclidean_distance(test_set[row1], training_set[row2]) #Calculating euclidean distance between one training set and all the test sets
            dist.append((d, label_training[row2]))

        sorted_distance = np.array(sorted(dist)) #Sorting the euclidean distance
        bingo = sorted_distance[:k[j], 1] #appending with K nearest neighbors 
        x,y = Counter(bingo).most_common(1)[0] #Deducting label based on maximum number of labels
        label_predicted.append(x)
    accuracy = (100*sum(label_test == label_predicted))/len(test_set) #Calculating the accuracy for given K values
    kAccuracy.append((accuracy, k[j]))
print kAccuracy
   
    

