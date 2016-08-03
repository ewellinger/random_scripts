import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Set style options
plt.style.use('ggplot')


def predicted_value_plot(model, df, column, classification=False, discrete_col=False, class_col=None, class_labels=None, bin_labels=None, freq=True, scatter=True, ci=True, scatter_num=None, fname=None, num_linspace=100, response_label=None, cmap=None, figsize=(10, 7)):
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


        class_labels: list, np.ndarray, or dict
        ci: bool
        class_col: int
        bin_labels: dict

    Possible parameters to add:
        - Ability to turn off confidence interval in the case of a continuous column
        - label list in the case of a multiclass classification
        - Ability to specify a particular class in a multiclass classification setting
    '''
    def _rand_jitter(arr, box):
        left_x, right_x = box.get_xdata()[0], box.get_xdata()[1]
        stdev = .18*(right_x - left_x)
        return arr + np.random.randn(len(arr)) * stdev

    def _discrete_value_plot(scatter_num, bin_labels):
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
            preds.append(np.array([boot_sample.mean() for boot_sample in (resample(pred) for _ in xrange(1000))]).reshape(1, -1))

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
        if not isinstance(bin_labels, dict):
            bin_labels = {}

        xtickNames = plt.setp(ax1, xticklabels=[bin_labels.get(label, label) for label in labels])
        plt.setp(xtickNames, rotation=-45, fontsize=10)

        # Superimpose jittered scatter plot if scatter is set to True
        if scatter:
            # If the number of points to plot was not set with scatter_num
            # Set the number to 125 * the number of discrete bins
            if not scatter_num:
                scatter_num = 125 * len(labels)
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
            lower = np.max([mean-1.5*std, df[column].min()])
            upper = np.min([mean+1.5*std, df[column].max()])
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
                pred = model.predict_proba(dfc)
                preds.append(np.array([boot_sample.mean(axis=0) for boot_sample in (resample(pred) for _ in xrange(1000))]).reshape(-1, pred.shape[1]))
            else:
                pred = model.predict(dfc)
                preds.append(np.array([boot_sample.mean() for boot_sample in (resample(pred) for _ in xrange(1000))]).flatten())

        probs = np.array(preds)

        # Subset down to a particular class if class_col is set
        if classification and class_col:
            probs = probs[:, :, class_col]

        prob_means = probs.mean(axis=1)
        lower_bounds = np.percentile(probs, q=10, axis=1)
        upper_bounds = np.percentile(probs, q=90, axis=1)

        # If probs.shape == 2 then we only need one line and confidence interval
        if len(probs.shape) == 2:
            # Create the fill to indicate the confidence bounds
            if ci:
                ax1.fill_between(x_i, lower_bounds, upper_bounds, facecolor=cmap[0], alpha=0.2)
            # Plot the predictions
            ax1.plot(x_i, prob_means, c=cmap[0], linewidth=2)
        else:
            for col_idx in xrange(probs.shape[2]):
                if ci:
                    ax1.fill_between(x_i, lower_bounds[:, col_idx], upper_bounds[:, col_idx], facecolor=cmap[col_idx], alpha=0.2)
                ax1.plot(x_i, prob_means[:, col_idx], c=cmap[col_idx], linewidth=2, label=class_labels.get(col_idx, col_idx))
                ax1.legend(loc='best')

        if freq:
            ax2 = ax1.twinx()

            # Set the color of the histogram to be distinct from any of the lines
            if len(probs.shape) == 2:
                freq_color = cmap[1]
            else:
                freq_color = cmap[probs.shape[2] + 1]

            if num_linspace:
                ax2.hist(df.loc[(df[column] >= mean-1.5*std) & (df[column] <= mean+1.5*std), column].values, facecolor=freq_color, alpha=0.4)
            else:
                ax2.hist(df[column].values, facecolor=freq_color, alpha=0.4)
            ax2.set_ylabel('Frequency')

        # Set xlims to mirror the min and max datapoint
        ax1.set_xlim([x_i.min(), x_i.max()])


    # Copy our DataFrame so as not to alter the original data
    dfc = df.copy()

    # If dealing with classification model set up class_labels
    if isinstance(class_labels, dict):
        pass
    elif isinstance(class_labels, (list, np.ndarray)):
        class_labels = {idx: label for idx, label in enumerate(class_labels)}
    else:
        class_labels = {}

    # Configure cmap if none is provided using Tableau's categorical colors
    if cmap == None:
        cmap = ['#4d79a8', '#e15759', '#f28e2b', '#76b7b2', '#59a14e', '#edc948', '#b07aa2', '#ff9da8', '#9c755f', '#bab0ac']

    # Create our Matplotlib figure and axis objects
    fig, ax1 = plt.subplots(figsize=figsize)

    if discrete_col:
        _discrete_value_plot(scatter_num, bin_labels)
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
    ''' Run predictd values plot on regression model '''
    # Read in diamonds dataset
    # See http://docs.ggplot2.org/0.9.3.1/diamonds.html for more info
    df = pd.read_csv('./toy_data/diamonds.csv')

    # Randomly sample 10000 datapoints to make confidence intervals more visible
    df = df.sample(n=10000, random_state=42)

    # Get dummies of some categorical columns
    df = pd.get_dummies(df, columns=['color', 'clarity'], drop_first=True)

    # Convert cut to int value
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    df['cut'] = df['cut'].map(lambda x: cut_map[x])

    # Make regression model using 'price' column as target
    y = df.pop('price').values
    X = df.values

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)

    # Run on a continous column
    predicted_value_plot(rf, df, 'carat', response_label="Predicted Price in U.S. Dollars", fname='./imgs/regression_predicted_value_plot')

    # Run on a discrete column
    bin_labels = {v: k for k, v in cut_map.iteritems()}
    predicted_value_plot(rf, df, 'cut', response_label="Predicted Price in U.S. Dollars", discrete_col=True, bin_labels=bin_labels, fname='./imgs/regression_discrete_predicted_value_plot')


    ''' Run predicted values plot on classification model '''

    # Reread in the dataframe
    df = pd.read_csv('./toy_data/diamonds.csv')

    # Subset down to 'Very Good' cut or above
    # df = df.loc[df['cut'].isin(['Ideal', 'Premium', 'Very Good']), :]

    # Map cut to integer label
    cut_map = cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    class_labels = {v: k for k, v in cut_map.iteritems()}
    df['cut'] = df['cut'].map(lambda x: cut_map[x])

    # Balance our classes so there are 2000 datapoints per cut class
    df = pd.concat([df.loc[df['cut'] == label].sample(n=2000, replace=True, random_state=42) for label in df['cut'].unique()], ignore_index=True)

    # Get dummies of some categorical columns
    df = pd.get_dummies(df, columns=['color', 'clarity'], drop_first=True)

    # Make classification model using 'cut' column as target
    y = df.pop('cut').values
    X = df.values

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    # Run on a continous column
    predicted_value_plot(rf, df, 'price', classification=True, class_labels=class_labels, fname='./imgs/classification_predicted_prob_plot')
