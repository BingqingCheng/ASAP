import numpy as np

def kernel_random_split(X,y,r=0.05):

    if (X.shape[0] != X.shape[1]):
        raise ValueError('K matrix is not a square')
    if (len(X) != len(y)):
        raise ValueError('Length of the vector of properties is not the same as number of samples')

    n_sample = len(X)
    all_list = np.arange(n_sample)
    randomchoice =  np.random.rand(n_sample)
    test_member_mask = (randomchoice < r)
    train_list = all_list[~test_member_mask]
    test_list = all_list[test_member_mask]

    X_train = X[:,train_list][train_list]
    y_train = y[train_list]

    X_test = X[:,train_list][test_list]
    y_test = y[test_list]
    return X_train, X_test, y_train, y_test



