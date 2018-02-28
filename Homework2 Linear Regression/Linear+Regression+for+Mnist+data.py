
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
Mnist_train= pd.read_csv('MNIST_training.csv')
Mnist_test = pd.read_csv('MNIST_test.csv')

test_data = Mnist_test.iloc[:,1:] #Test Data without Label
train_data = Mnist_train.iloc[:,1:] #Train Data without Label
train_label = Mnist_train.iloc[:,0] #Train Data Label
test_label = Mnist_test.iloc[:,0] #Test Data Label

#Normalisation of Data
test_data = test_data / 255
train_data = train_data / 255

#Adding column of ones in first olumn
test_data.insert(0,'Ones',1)
train_data.insert(0,'Ones',1)

#Convert type to array
X = np.array(train_data)
Y = np.array(train_label)
test_label = np.array(test_label)
test_data = np.array(test_data)

X_tran = X.T #Transpose of X
A = np.matmul(X_tran, X) #X'X
A_inv = np.linalg.pinv(A) #(A)-'
B = np.matmul(X_tran, Y) #X'Y

coeff = np.matmul(A_inv, B) #Optimal value of b
y_pred = np.matmul(test_data, coeff) #Y = Xb


# In[55]:


#Calculation of Accuracy with a threshold value of 0.5
threshold = sum([y_pred > 0.5])
#threshold
accuracy_linear = sum(threshold == test_label)
accuracyLinear_percentage = ((float(accuracy_linear) / len(test_label) * 100))
accuracyLinear_percentage


# In[70]:


#Gradient Descent
def gradient_descent(traindata, label, coeff):
    descent = np.dot(np.dot(traindata.transpose(), traindata), coeff) - np.dot(traindata.transpose(), label) #Descent Formula - First Derivate
    return descent
coeff_descent = np.zeros(train_data.shape[1]) #Initial Coefficients to Zero

#OlS cost function to minimise the error
def cost_function(traindata, label, coeff):
    cost = np.sum((np.dot(traindata, coeff) - np.array(label))**2) #Optimization Formula
    return cost
cost_value=[]
learning_rate = 1e-4  #0.00001 for getting 100% accuracy

#Updating the Weights
for i in range(0, 100):
    coeff_descent = coeff_descent - learning_rate * gradient_descent(X, Y, coeff_descent)
    cost = cost_function(X, Y, coeff_descent)
    cost_value.append(cost)


# In[72]:


import matplotlib.pyplot as plt
plt.plot(cost_value)
plt.show()

y_predict = np.matmul(test_data, coeff_descent) 
threshold_descent = sum([y_predict>0.5]) #Threshold value of 0.5

accuracy_descent = sum(threshold_descent == test_label)
accuracyDescent_percentage = ((float(accuracy_descent)/len(test_label)*100))
print accuracyDescent_percentage


print(sum(abs(coeff_descent - coeff)))

