# Python-for-Machine-Learning
Machine Learning Assignments
KNN :
For this homework assignment, please follow the procedure:
1. For each data in “MNIST_test.csv”, compute distances with the training data.
2. Find the K-nearest neighbors, and decide the majority class of them.
3. Compare the prediction with the ground truth
a. Correctly classified if the predicted label and ground truth is identical.
b. Incorrectly classified if the predicted label and ground truth is NOT identical.
4. Repeat Step 1-4 for all data in the test data
5. Then, you can count how many test data are correctly classified and incorrectly classified.
6. Show the accuracy of your KNN. Compute accuracy by:
𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦= # 𝑜𝑓 𝑦𝑜𝑢𝑟 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑙𝑦 𝑐𝑙𝑎𝑠𝑠𝑖𝑓𝑖𝑒𝑑 / # 𝑜𝑓 𝑡𝑜𝑡𝑎𝑙 𝑡𝑒𝑠𝑡 𝑑𝑎𝑡𝑎
You CANNOT use any libraries or built-in functions of KNN. You have to implement it.


Linear Regression: 
Task 1. Classifying MNIST data
Download the MNIST data in the course web page. There are two files: MNIST_training.csv and MNIST_test.csv. Each dataset includes 200 image samples for label of 0 and 1 among ten labels (0-9). So, we will solve a binary classification problem.
You will train a linear regression model using the training data (MNIST_training.csv) and will compute accuracy with the test data (MNIST_test.csv).
For Task 1, please follow the procedure:
1. Train a linear regression model
- Find the optimal coefficients (b_opt) by the equation: b_opt = (X'X)^(-1)X'y
- Please use "numpy.linalg.pinv" for matrix inverse. The function computes pseudo-inverse. If you use "numpy.linalg.inv", it will cause an error.
2. Display the optimal coefficients (denoted by b_opt)
3. Classify test data (MNIST_test.csv) with a threshold of 0.5.
- y_pred = X_test * b_opt
- if y_pred > 0.5, class 1, otherwise 0
4. Display the accuracy
Task 2. Implementation of Gradient Descent with MNIST data
For Task 2, we will use the same data as Task 1. However, we will find the optimal coefficients by using "Gradient Descent" algorithm. Then, we will compare with the solution that we found in Task 1.
The procedure of Task 2 is the almost same as Task 1, but need to implement "Gradient Descent" algorithm, instead of a single line equation ((X'X)^(-1)X'y).
For the Gradient Descent algorithm, please follow the procedure:
1. Set the initial coefficient to zeros (can be any random values though)
- Think of what the dimension of the coefficient vector is
2. Determine hyper-parameters such as learning rate and iteration numbers
3. Run "gradient descent" algorithm with the hyper-parameters and check "Learning Curve"
* Learning curve shows whether it converges or not. X-axis shows iteration, while y-axis shows
cost
* Learning curve has to be shown as "converged", otherwise the solution may be not good.
4. Display the optimal coefficients (denoted by b_est)
5. Classify test data (MNIST_test.csv) with a threshold of 0.5.
- y_pred = X_test * b_est
- if y_pred > 0.5, class 1, otherwise 0
6. Display the accuracy
7. Display the total differences between b_opt and b_est
- sum(abs(b_opt – b_est))
