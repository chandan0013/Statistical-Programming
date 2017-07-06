
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

np.random.seed(1)

def accuracy(p, y):
   
    y = 2 * y - 1
    return np.mean((p > 0) == (y == 1))



def train(X, y,num_iterations, learning_rate, lamb):
    
    n, p = X.shape
    p = p + 1
    
    X1 = np.hstack((np.ones((n,1)),X)) 
    y = 2 * y - 1

    beta = np.zeros((p,1))

    for it in range(num_iterations):
        s = np.dot(X1, beta)
        db = np.multiply(s,y) < 1
        blah = np.repeat(np.multiply(db,y), p, axis = 1)
        dbeta = np.dot(np.ones((1,n)), np.multiply(blah, X1/n)) 
        beta = beta + learning_rate * np.transpose(dbeta)
        beta[1:p] = beta[1:p] + lamb * beta[1:p]

    return beta 



def test(X, model):
    
    n,c = X.shape
    X1 = np.hstack((np.ones((n,1)),X)) 
    y_predict = np.dot(X1,model)
    return y_predict



def load_digits(subset=None, normalize=True):
    """
    Load digits and labels from digits.csv.

    Args:
        subset: A subset of digit from 0 to 9 to return.
                If not specified, all digits will be returned.
        normalize: Whether to normalize data values to between 0 and 1.

    Returns:
        digits: Digits data matrix of the subset specified.
                The shape is (n, p), where
                    n is the number of examples,
                    p is the dimension of features.
        labels: Labels of the digits in an (n, ) array.
                Each of label[i] is the label for data[i, :]
    """
    # load digits.csv, adopted from sklearn.
    import pandas as pd
    df = pd.read_csv('digits.csv')

    # only keep the numbers we want.
    if subset is not None:
        df = df[df.iloc[:,-1].isin(subset)]

    # convert to numpy arrays.
    digits = df.iloc[:,:-1].values.astype('float')
    labels = df.iloc[:,-1].values.astype('int')

    # Normalize digit values to 0 and 1.
    if normalize:
        digits -= digits.min()
        digits /= digits.max()

    # Change the labels to 0 and 1.
    for i in xrange(len(subset)):
        labels[labels == subset[i]] = i

    labels = labels.reshape((labels.shape[0], 1))
    return digits, labels



def split_samples(digits, labels):
    """Split the data into a training set (70%) and a testing set (30%)."""
    num_samples = digits.shape[0]
    num_training = round(num_samples * 0.7)
    indices = np.random.permutation(num_samples)
    training_idx, testing_idx = indices[:num_training], indices[num_training:]
    return (digits[training_idx], labels[training_idx],
            digits[testing_idx], labels[testing_idx])


#====================================
# Load digits and labels.
digits, labels = load_digits(subset=[3, 5], normalize=True)
training_digits, training_labels, testing_digits, testing_labels = split_samples(digits, labels)
print '# training', training_digits.shape[0]
print '# testing', testing_digits.shape[0]


#====================================
# Train SVM and display training accuracy.

def mySVM(training_digits, training_labels, testing_digits, testing_labels):

    regularization_param = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])
    training_accuracy_reg = np.zeros(regularization_param.shape)
    testing_accuracy_reg = np.zeros(regularization_param.shape)
    regularization_param_log = np.zeros(8)

    for it in range(0,len(regularization_param)):
        regularization_param_log[it] = math.log(regularization_param[it])
        model = train(training_digits, training_labels, 200, 0.01, regularization_param[it])
    
        # Evaluate on the training set
        y_predict_train = test(training_digits, model)
        training_accuracy_reg[it] = accuracy(y_predict_train, training_labels)
        print 'Accuracy on training data for regularization value =  %.3f : %.4f' % (regularization_param[it], training_accuracy_reg[it])

        # Evaluate on the testing set.
        y_predict_test = test(testing_digits, model)
        testing_accuracy_reg[it] = accuracy(y_predict_test, testing_labels)
        print 'Accuracy on testing data for regularization value = %.3f : %.4f' % (regularization_param[it], testing_accuracy_reg[it])


    #====================================
    #Train SVM and display training accuracy.
    learning_rate = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])
    training_accuracy_lr = np.zeros(learning_rate.shape)
    testing_accuracy_lr = np.zeros(learning_rate.shape)
    learning_rate_log = np.zeros(learning_rate.shape)

    for it in range(0,len(learning_rate)):
        learning_rate_log[it] = math.log(learning_rate[it])
        model = train(training_digits, training_labels, 50, learning_rate[it], 0.05)
    
        # Evaluate on the training set
        y_predict_train = test(training_digits, model)
        training_accuracy_lr[it] = accuracy(y_predict_train, training_labels)
        print 'Accuracy on training data for learning_rate =  %.3f : %.4f' % (learning_rate[it], training_accuracy_lr[it])

        # Evaluate on the testing set.
        y_predict_test = test(testing_digits, model)
        testing_accuracy_lr[it] = accuracy(y_predict_test, testing_labels)
        print 'Accuracy on testing data for learning_rate = %.3f : %.4f' % (learning_rate[it], testing_accuracy_lr[it])



    plt.figure(figsize=(10,5))

    plt.subplot(2, 2, 1)
    plt.ylabel('Training Accuracy')
    plt.xlabel('Log(Regularization Parameter)')
    plt.plot(regularization_param_log, training_accuracy_reg)

    plt.subplot(2, 2, 2)
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Log(Regularization Parameter)')
    plt.plot(regularization_param_log, testing_accuracy_reg)

    plt.subplot(2, 2, 3)
    plt.ylabel('Training Accuracy')
    plt.xlabel('Log(Learning Rate)')
    plt.plot(learning_rate_log, training_accuracy_lr)

    plt.subplot(2, 2, 4)
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Log(Learning Rate)')
    plt.plot(learning_rate_log, testing_accuracy_lr)

    plt.show()

mySVM(training_digits, training_labels, testing_digits, testing_labels)
