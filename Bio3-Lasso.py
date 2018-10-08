
# coding: utf-8

# In[167]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
from sklearn import linear_model

#TASK 1
train_gene=pd.read_csv('C:\Users\uboyanap\Dropbox\Share\Bioinformatics\Home work 3\Gene_expression_1.csv',header=None,)
corr=train_gene.corr()
corr_square=corr**2
pair_corr=np.subtract(corr_square, np.identity(10))
print pair_corr #Computing Pair wise Correlation Matrix
threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
adj_mat = pd.read_csv('C:\Users\uboyanap\Dropbox\Share\Bioinformatics\Home work 3\Adj_1.csv', header=None)
bool_mat = adj_mat.astype(np.int)
#print bool_mat
x_vector=[]
y_vector=[]
for t in threshold:
    #Comparing adjacency matrix between the Network and Ground truth
    compare_zero=np.zeros(shape=(10,10))
    for i in range(len(pair_corr)):
        for j in range(i+1, len(pair_corr)):
            if pair_corr[i][j] <= t:
                compare_zero[i][j] = 0
                compare_zero[j][i] = 0
            else:
                compare_zero[i][j] = 1
                compare_zero[j][i] = 1
    
    TP=0
    TN=0
    FP=0
    FN=0
    #Compute Confusion Matrix
    for u in range(len(compare_zero)):
        for v in range(len(compare_zero)):
            if compare_zero[u][v]==1 and bool_mat[u][v]==1:
                TP +=1
            elif compare_zero[u][v]==0 and bool_mat[u][v]==0:
                TN +=1
            elif compare_zero[u][v]==1 and bool_mat[u][v]==0:
                FP +=1
            elif compare_zero[u][v]==0 and bool_mat[u][v]==1:
                FN +=1            
    TPR= TP/(TP+FN) #Computing TPR for Threshold
    y_vector.append(TPR)
    FPR =FP/(FP+TN)
    x_vector.append(FPR) #Computing FPR for Threshold 
    print("TPR: ", TPR)
    print("FPR: ", FPR)
#ROC Plot
x=x_vector
y=y_vector
plt.plot(x,y, color='blue', marker='p')
plt.axis([-0.1, 1, 0, 1.1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

#TASK 2

lasso_train = pd.read_csv('C:\Users\uboyanap\Dropbox\Share\Bioinformatics\Home work 3\Gene_expression_2.csv',sep=',',header=None,)
lasso_adj = pd.read_csv('C:\Users\uboyanap\Dropbox\Share\Bioinformatics\Home work 3\Adj_2.csv',sep=',',header=None)
lasso_train = np.array(lasso_train)
lamb_das = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.05, 0.1, 0.5, 1, 10, 100]
x_vector2=[]
y_vector2=[]
for values in lamb_das:  #Constructing adjacency matrix with result of Lasso
    adj_matrix = []
    compare_matrix = []
    clf=linear_model.Lasso(values)
    for i in range(len(lasso_train.T)): 
        #print i  
        y=lasso_train[:,i]
        condition = np.array(lasso_train, dtype=bool)
        condition[:,i]=False
        x=np.extract(condition, lasso_train)
        x=np.reshape(x, (500,9))
        #print x
        clf.fit(x,y)
        #print clf.coef_
        row = np.insert(clf.coef_, i, 0)
        adj_matrix += [row]
        compare_y = []
        for comp in row:
            if comp == 0:
                compare_y += [0]
            else:
                compare_y += [1]
        compare_matrix += [compare_y]
        #print adj_matrix     
        
    TP=0
    TN=0
    FP=0
    FN=0
    #Compute Confusion Matrix 
    for u in range(len(compare_matrix)):
        for v in range(len(compare_matrix)):
            if compare_matrix[u][v]==1 and lasso_adj[u][v]==1:
                TP +=1
            elif compare_matrix[u][v]==0 and lasso_adj[u][v]==0:
                TN +=1
            elif compare_matrix[u][v]==1 and lasso_adj[u][v]==0:
                FP +=1
            elif compare_matrix[u][v]==0 and lasso_adj[u][v]==1:
                FN +=1            
    TPR2= TP/(TP+FN) #Computing TPR for Threshold
    y_vector2.append(TPR2)
    FPR2 =FP/(FP+TN)
    x_vector2.append(FPR2) #Computing FPR for Threshold    
    
    print ("TPR2: ", TPR2)
    print ("FPR2: ", FPR2)

x=x_vector2
y=y_vector2
plt.plot(x,y, color='blue', marker='p')
plt.axis([-0.1, 1, 0, 1.1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

#print lasso_train

#With Library
#for 






# In[173]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
lasso_result=[]
lasso_matrix=[]

def fit(X,Y):
    a = np.dot(X.T, X)
    b = np.dot(X.T, Y)
    a_inverse = np.linalg.inv(a)
    coefficient = np.dot(a_inverse, b)
    #print ("Coeff", coefficient)
    return coefficient

for i in range (len(lasso_train.T)):
    if i==0:
        X=lasso_train[:,1:10]
        #print X
        #print("X: ", X.shape)
        X1 = np.insert(X, 0, values=1, axis=1)
        Y=lasso_train[:,i]
        lasso_result.append(fit(X1,Y))
        lasso_result= [row[1:] for row in lasso_result]
        first_row=np.insert(lasso_result,0,values=0,axis=1)
        print(first_row.shape)
        
    elif 0< i <9:
        X1=lasso_train[:,0:i]
        X2=lasso_train[:,i+1:10]
        X=np.concatenate((X1,X2),axis=1)
        X1=np.insert(X,0,values=1,axis=1)
        Y=lasso_train[:,i]
        P=fit(X1,Y)
        m=np.delete(P,0)
        j=m[:i]
        k=m[i:,]
        Matrix=np.append(j,[0])
        Matrix=np.append(Matrix,k)
        lasso_matrix.append(Matrix)
    else:
        X=lasso_train[:,0:9]
        X1=np.insert(X,0,values=1,axis=1)
        Y=lasso_train[:,i]
        Z=fit(X1,Y)
        Z=np.delete(Z,0)
        Z=np.append(Z,[0])
        #print("Z",Z)
        
adj_matrix=np.vstack([first_row, lasso_matrix])
adj_matrix=np.vstack([adj_matrix, Z])

#print ("Ajacency Matrix: ", adj_matrix)
fpr_x=[]
tpr_y=[]
lamb_das = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.05, 0.1, 0.5, 1, 10, 100]
#print ("Actual Matrix", actual)
for i in lamb_das:
    h= adj_matrix>i
    h=1*h
    lasso_prediction = np.array(h)
    actual = np.array(lasso_adj)
    #print ("Adjacency Matrix: ", i,"\n",h)
    TP = np.sum(np.logical_and(lasso_prediction == 1, actual == 1))
    TN = np.sum(np.logical_and(lasso_prediction == 0, actual == 0))
    FP = np.sum(np.logical_and(lasso_prediction == 1, actual == 0))
    FN = np.sum(np.logical_and(lasso_prediction == 0, actual == 1))
    M = [[0 for y in range(2)] for x in range(2)]
    M[0][0] = TP
    M[0][1] = FP
    M[1][0] = FN
    M[1][1] = TN
    #print("Confusion Matrix: ",i,"\n", np.matrix(M))
    #print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    TPR= TP/(TP+FN) #Computing TPR for Threshold
    tpr_y.append(TPR)
    FPR =FP/(FP+TN)
    fpr_x.append(FPR) #Computing FPR for Threshold 
    print ("TPR:", TPR)
    print ("FPR:", FPR)

#ROC Plot
x=fpr_x
y=tpr_y
plt.plot(x,y, color='blue', marker='p')
plt.axis([-0.1, 1.2, -0.1, 1.2])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(x,y)
plt.show()






# In[ ]:



