import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def permutation_significance(model, X, y, X_idx=None, num=1000):
    '''
    Returns the corresponding features null coefficents calculated by
        permutation

    Parameters
    -----------
    model : model that implements the model.fit() method and model.coef_
        attribute
    X : numpy array containing our feature matrix
    y : numpy array of targets
    X_idx : int (default None)
        int indicating the index of the column in question.  If None, will
        return an array of all permuted coefficients
    num : int (default 1000)
        Number of times to run the permutation

    Returns
    --------
    arr : Numpy array containing the null coefficients
    '''
    np.random.seed(1234)
    yc = y.copy()
    coefs = []
    for _ in range(num):
        np.random.shuffle(yc)
        model.fit(X, yc)
        coefs.append(model.coef_)
    return np.array(coefs)[:, X_idx] if X_idx is not None else np.array(coefs)

def permutation_significance_plot(model, X, y, X_idx, label=None):
    '''
    Create a permutation plot and calculates a p-value for a particular
        coefficent

    Parameters
    -----------
    model : A fitted regression model that implements the model.coef_ attribute
    X : numpy array containing our feature matrix
    y : numpy array of targets
    X_idx : int indicating the index of the column in question
    label : str for labeling the xaxis of the plot

    Returns
    --------
    ax : Matplotlib axis object
    '''
    original_coef = model.coef_[X_idx]
    permuted_coef = permutation_significance(model, X, y, X_idx)

    prob = (np.absolute(permuted_coef) >= abs(original_coef)).mean()

    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(permuted_coef, bins=30, color='k', alpha=0.3, label='Null')
    plt.axvline(x=original_coef, color='r', linestyle='-', label='Estimate', linewidth=3)
    plt.legend(loc='best')
    plt.title('Permuted Coefficient Significance Plot')
    plt.xlabel(r"{}: $\beta$={:.3}, $p$={:.3}".format(label, original_coef, prob))
    return ax
