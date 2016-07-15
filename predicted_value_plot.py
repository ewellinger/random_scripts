import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import resample
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
matplotlib.style.use('ggplot')


def predicted_value_plot(model, df, column, classification=False, fname=None, num_linspace=100, response_label=None, figsize=(8, 6)):
    ''' Makes predicted values/probability plot for a given feature
    INPUT:
        model: Trained model that implements a .predict() or, in the case of classification problems, .predict_proba() methods
        df: Pandas Dataframe
            Dataframe containing only the features
        column: str
            String indicating the column of the df to make the plot for
        classification: bool, default False
            Indicates whether model is a classification or regression type
        fname: str or bool, default None
            Indicates whether to save the resulting figure.  If None or False the figure will simply be created.  True will save the figure with the file name 'predicted_value_{column}.png'.  str input will save the figure with the file name '{str}.png'
        num_linspace: int or None, default 100
            Determines how finely we will step through the values of our feature.  Setting this to None or False will result in iterating through the unique values of the feature.
            NOTE: This should be set to None if the column in question is a discrete variable!
        response_label: str, default None
            Label for y-axis
        figsize: tuple (int, int)
            Size of figure in inches
    '''
    # We need to set up an array of x_i values to search over
    if num_linspace:
        # Create an interval over +-1 std of the mean of the x column
        mean, std = df[column].mean(), df[column].std()
        lower = np.max([mean-std, df[column].min()])
        upper = np.min([mean+std, df[column].max()])
        x_i = np.linspace(lower, upper, num=num_linspace)
    else:
        # If num_linspace=None, make x_i the unique values
        x_i = np.unique(df[column])

    # Copy our DataFrame so as not to alter the original data
    dfc = df.copy()

    # For each value in our search space, set the entire column in question to that value and run model.predict or model.predict_proba
    # Average out those predictions and add it to a list of y_hats that we are keeping track of
    preds = []
    for val in x_i:
        dfc[column] = val
        if classification:
            pred = model.predict_proba(dfc)[:, 1]
        else:
            pred = model.predict(dfc)

        preds.append([boot_sample.mean() for boot_sample in (resample(pred) for _ in xrange(1000))])
    probs = np.array(preds)
    prob_means = probs.mean(axis=1)
    lower_bounds = np.percentile(probs, q=10, axis=1)
    upper_bounds = np.percentile(probs, q=90, axis=1)

    # Create our Matplotlib figure and axis objects
    fig, ax1 = plt.subplots(figsize=figsize)

    if response_label:
        ax1.set_ylabel(response_label)
    elif classification:
        ax1.set_ylabel('Predicted Probability ($\hat{\pi}$)')
    else:
        ax1.set_ylabel('Predicted Response ($\hat{y}$)')
    ax1.set_xlabel('{}'.format(column), fontsize=14)
    plt.title('Predicted Values Plot for {}'.format(column))

    # Create the fill to indicate the confidence bounds
    ax1.fill_between(x_i, lower_bounds, upper_bounds, alpha=0.25)
    ax1.plot(x_i, prob_means, linewidth=2)

    ax2 = ax1.twinx()
    if num_linspace:
        ax2.hist(df.loc[(df[column] >= mean-std) & (df[column] <= mean+std), column].values, alpha=0.4)
    else:
        ax2.hist(df[column].values, alpha=0.4)
    ax2.set_ylabel('Frequency')

    plt.tight_layout()

    if fname:
        if fname == True:
            plt.savefig('./partial_dependency_{}.png'.format(column), dpi=300)
        else:
            plt.savefig('{}.png'.format(fname), dpi=300)


if __name__=='__main__':
    boston = load_boston()
    y = boston.target  # House prices
    X = boston.data  # The other 13 features
    col_names = boston.feature_names
    df = pd.DataFrame(X, columns=col_names)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)

    predicted_value_plot(rf, df, 'CRIM', response_label="Predicted Median Value of Homes in $1,000's", fname='./imgs/regression_predicted_value_plot')
