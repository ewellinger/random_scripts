import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib
from cycler import cycler
from sklearn.utils import resample
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Set style options
plt.style.use('ggplot')
color_cycle = cycler(color=['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd'])
matplotlib.rcParams['axes.prop_cycle'] = color_cycle


def predicted_value_plot(model, df, column, classification=False, discrete_col=False, freq=True, scatter=True, scatter_num=None, fname=None, num_linspace=100, response_label=None, cmap=('#4c72b0', '#c44e52'), figsize=(8, 6)):
    ''' Makes predicted values/probability plot for a given feature
    INPUT:
        model: Trained model that implements a .predict() or, in the case of classification problems, .predict_proba() methods
        df: Pandas Dataframe
            Dataframe containing only the features
        column: str
            String indicating the column of the df to make the plot for
        classification: bool, default False
            Indicates whether model is a classification or regression type via a True or False value respectively
        discrete_col: bool, default False
            Indicates whether the column in question is a continuous or discrete variable.  If discrete, predicted box plots are made for each category.
        freq: bool, default True
            Indicates whether to superimpose a histogram over the predicted values plot (NOTE: this only occurs if discrete_col=False)
        scatter: bool, default True
            Indicates whether to superimpose a jittered scatter plot of bootstrapped means over each categories boxplot (NOTE: this only occurs if discrete_col=True)
        scatter_num: int, default None
            Dictates how many points to plot when discrete_col=True and scatter=True.  scatter_num points are distributed across the bins according to their original distribution.
            If None, 200 * number of categories of points are plotted.
        fname: str or bool, default None
            Indicates whether to save the resulting figure.  If None or False the figure will simply be created.  True will save the figure with the file name 'predicted_value_{column}.png'.  str input will save the figure with the file name '{str}.png'
        num_linspace: int or None, default 100
            Determines how finely we will step through the values of our feature.  Setting this to None or False will result in iterating through the unique values of the feature.
            NOTE: This should be set to None if the column in question is a discrete variable!
        response_label: str, default None
            Label for y-axis
        cmap: tuple, (primary-color, secondary-color)
            Tuple of colors dictating the primary and secondary colors of the plot.  Any Matplotlib accepted color values can be passed.
        figsize: tuple (int, int)
            Size of figure in inches
    '''
    def _rand_jitter(arr, box):
        left_x, right_x = box.get_xdata()[0], box.get_xdata()[1]
        stdev = .25*(right_x - left_x)
        return arr + np.random.randn(len(arr)) * stdev

    def _discrete_value_plot(scatter_num):
        # Create an array of the unique discrete bins
        labels = np.unique(dfc[column])

        # Create list for keeping track of predictions
        preds = []
        for label in labels:
            # Set all of that column to that particular label
            dfc[column] = label
            # Make predictions using inputed model
            if classification:
                pred = model.predict_proba(dfc)[:, 1]
            else:
                pred = model.predict(dfc)

            # Append array of means of bootstrapped predictions
            preds.append(np.array([boot_sample.mean() for boot_sample in (resample(pred) for _ in xrange(1000))]).reshape(-1, 1))

        # Probably do this irrespective of discrete vs contin
        # fig, ax1 = plt.subplots(figsize=figsize)

        # Create the boxplots for each label and alter colors
        bp = plt.boxplot(preds, sym='', whis=[5,95], labels=labels) #, widths=0.35)
        plt.setp(bp['boxes'], color=cmap[0])
        plt.setp(bp['whiskers'], color=cmap[0])
        plt.setp(bp['caps'], color=cmap[0])

        # Fill the boxes with color
        for idx in xrange(len(labels)):
            box = bp['boxes'][idx]
            boxCoords = box.get_xydata()
            boxPolygon = Polygon(boxCoords, facecolor=cmap[0], alpha=0.7)
            ax1.add_patch(boxPolygon)

        # Set the xtick labels
        xtickNames = plt.setp(ax1, xticklabels=labels)
        plt.setp(xtickNames, rotation=-45, fontsize=10)

        # Superimpose jittered scatter plot if scatter is set to True
        if scatter:
            # If the number of points to plot was not set with scatter_num
            # Set the number to 200 * the number of discrete bins
            if not scatter_num:
                scatter_num = 200 * len(labels)
            # Make the number of points per bin perportional to that labels
            # representation in the original dataset
            num_per_label = [int((df[column] == label).mean() * scatter_num) for label in labels]
            for idx, num in enumerate(num_per_label):
                y_data = np.random.choice(preds[idx].flatten(), size=num)
                x_data = [bp['whiskers'][idx*2].get_xdata()[0]] * num
                jittered_x = _rand_jitter(x_data, bp['boxes'][idx])
                ax1.scatter(jittered_x, y_data, c=cmap[1], alpha=0.6)

    def _contin_value_plot():
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

        # Create the fill to indicate the confidence bounds
        ax1.fill_between(x_i, lower_bounds, upper_bounds, facecolor=cmap[0], alpha=0.25)
        # Plot the predictions
        ax1.plot(x_i, prob_means, c=cmap[1], linewidth=2)

        if freq:
            ax2 = ax1.twinx()
            if num_linspace:
                ax2.hist(df.loc[(df[column] >= mean-std) & (df[column] <= mean+std), column].values, facecolor=cmap[1], alpha=0.4)
            else:
                ax2.hist(df[column].values, facecolor=cmap[1], alpha=0.4)
            ax2.set_ylabel('Frequency')

    # Copy our DataFrame so as not to alter the original data
    dfc = df.copy()

    # Create our Matplotlib figure and axis objects
    fig, ax1 = plt.subplots(figsize=figsize)

    if discrete_col:
        _discrete_value_plot(scatter_num)
    else:
        _contin_value_plot()

    if response_label:
        ax1.set_ylabel(response_label)
    elif classification:
        ax1.set_ylabel('Predicted Probability ($\hat{\pi}$)')
    else:
        ax1.set_ylabel('Predicted Response ($\hat{y}$)')
    ax1.set_xlabel('{}'.format(column), fontsize=14)
    plt.title('Predicted Values Plot for {}'.format(column))

    plt.tight_layout()

    # Save the figure
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

    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X, y)

    predicted_value_plot(rf, df, 'CRIM', response_label="Predicted Median Value of Homes in $1,000's", fname='./imgs/regression_predicted_value_plot')
    plt.clf()

    predicted_value_plot(rf, df, 'CHAS', response_label="Predicted Median Value of Homes in $1,000's", discrete_col=True)
    plt.xlabel('Charles River Property')
    plt.savefig('./imgs/regression_discrete_predicted_value_plot.png', dpi=300)


    # # Plot of regression model for continuous column with frequency overlaid
    # predicted_value_plot(rf, df, 'CRIM')
    #
    # # Plot of regression model for continuous column w/o frequency overlaid
    # predicted_value_plot(rf, df, 'CRIM', freq=False)
    #
    # # Plot of regression model for discrete column with scatter overlaid
    # predicted_value_plot(rf, df, 'CHAS', discrete_col=True)
    #
    # # Plot of regression model for discrete column w/o scatter overlaid
    # predicted_value_plot(rf, df, 'CHAS', discrete_col=True, scatter=False)
