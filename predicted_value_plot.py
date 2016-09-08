import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import collections

# Set style options
plt.style.use('ggplot')


def predicted_value_plot(model, df, column, classification=False, discrete_col=False, class_col=None, class_labels=None, bin_labels=None, freq=True, scatter=True, ci=(5, 95), scatter_num=None, num_linspace=100, response_label=None, cmap=None, figsize=(10, 7)):
    ''' Makes predicted values/probability plot for a given feature
    INPUT:
        model: Obj
            Trained model that implements a .predict() or, in the case of
            classification problems, .predict_proba() method
        df: Pandas Dataframe
            Dataframe containing only the features
        column: str
            String indicating the column of the df to make the plot for
        classification: bool, default False
            Indicates whether model is a classification or regression type via
            a True or False value respectively
        discrete_col: bool, default False
            Indicates whether the column in question is a continuous or
            discrete variable.  If discrete, predicted box plots are made for
            each category.

        class_col: int, default None

        class_labels: something, default None

        bin_labels: something, default None

        freq: bool, default True
            Indicates whether to superimpose a histogram over the predicted
            values plot (NOTE: this only occurs if discrete_col=False)
        scatter: bool, default True
            Indicates whether to superimpose a jittered scatter plot of
            bootstrapped means over each categories boxplot (NOTE: this only
            occurs if discrete_col=True)

        ci: tuple or False, default=(5, 95)
            Tuple indicating what confidence interval to convey.  Defaults to a
            90% confidence interval.  Set to False to turn off any confidence
            intervals (or whiskers in the case of box plots).
        scatter_num: int, default None
            Dictates how many points to plot when discrete_col=True and
            scatter=True.  scatter_num points are distributed across the bins
            according to their original distribution.
            If None, 125 * number of categories of points are plotted.
        num_linspace: int or None, default 100
            Determines how finely to step through the values of our feature.
            Setting this to None or False will result in iterating through the
            unique values of the feature. (NOTE: this only occurs if
            classification=False)
        response_label: str, default None
            Label for y-axis
        cmap: iterable of valid Matplotlib colors, default None
            If left to None or set to a non-iterable object, a cmap will be
            initialized to a list of 10 colors using Tableau's categorical
            colors.
            Values must be a Matplotlib accepted color.
        figsize: tuple (int, int)
            Size of figure in inches
    '''
    def _validate_inputs(class_labels, cmap):
        # Verify that model implements .predict()/.predict_proba()
        if classification and not hasattr(model, 'predict_proba'):
            raise TypeError(
                "'model' object must implement the predict_proba() method if classification=True")
        elif not hasattr(model, 'predict'):
            raise TypeError(
                "'model' object must implement the predict() method")

        if column not in df.columns:
            raise ValueError(
                "'column' must be a column in the passed DataFrame")

        # If set, verify class_col is an integer
        if not isinstance(class_col, (type(None), int)):
            raise ValueError(
                "arg 'class_col' must be an int, leaving class_col=None causes all classes to be included")

        # If dealing with classification model set up class_labels
        if isinstance(class_labels, dict):
            pass
        elif isinstance(class_labels, collections.Iterable):
            class_labels = {idx: label for idx,
                            label in enumerate(class_labels)}
        else:
            class_labels = {}

        # Configure cmap if none is provided using Tableau's categorical colors
        # To properly override these colors you must pass an iterable with
        # valid matplotlib color values
        if not isinstance(cmap, collections.Iterable):
            cmap = ['#4d79a8', '#e15759', '#f28e2b', '#76b7b2', '#59a14e',
                    '#edc948', '#b07aa2', '#ff9da8', '#9c755f', '#bab0ac']

        return class_labels, cmap

    def _rand_jitter(arr, box):
        left_x, right_x = box.get_xdata()[0], box.get_xdata()[1]
        stdev = .17 * (right_x - left_x)
        return arr + np.random.randn(len(arr)) * stdev

    def _get_xtick_loc(num_classes, num_bins):
        num_boxplots = (num_classes * num_bins) + num_bins - 1
        return np.arange(.5 + (float(num_classes) / 2), num_boxplots + 1, num_classes + 1)

    def _discrete_value_plot(scatter_num, bin_labels):
        # Create an array of the unique discrete bins
        labels = np.unique(dfc[column])

        # Set the xtick labels
        # If a dictionary wasn't passed in for the bin_labels, set each to that
        # label's value
        if not isinstance(bin_labels, dict):
            bin_labels = {}

        # Create list for keeping track of predictions
        preds = []
        for label in labels:
            # Set all of that column to that particular label
            dfc[column] = label
            # Make predictions using inputed model
            if classification:
                pred = model.predict_proba(dfc)
                preds.append(np.array([boot_sample.mean(axis=0) for boot_sample in (
                    resample(pred) for _ in xrange(1000))]).reshape(-1, pred.shape[1]))
            else:
                pred = model.predict(dfc)
                preds.append(np.array([boot_sample.mean(axis=0) for boot_sample in (
                    resample(pred) for _ in xrange(1000))]).flatten())

        preds = np.array(preds)

        # Subset down to a particular class if class_col is set
        if classification and class_col:
            preds = preds[:, :, class_col]

        # If scatter is set to True, set up variable that will dictate the
        # number of points in a particular scatter plot which will be used
        # below
        if scatter:
            # If the number of points to plot was not set with scatter_num
            # Set the number to 125 * the number of discrete bins
            if not scatter_num:
                scatter_num = 125 * len(labels)
            # Make the number of points per bin perportional to that labels
            # representation in the original dataset
            num_per_label = [
                int((df[column] == label).mean() * scatter_num) for label in labels]

        # We must now plot each individual bin/class combo
        if classification and not class_col:
            num_classes = preds.shape[2]
            num_boxplots = (num_classes * len(labels)) + len(labels) - 1
            for class_num in xrange(num_classes):
                bp_positions = range(
                    class_num + 1, num_boxplots + 1, num_classes + 1)

                # plt.boxplot only likes getting a list of arrays
                data = [preds[idx, :, class_num]
                        for idx in range(preds.shape[0])]
                bp = plt.boxplot(data, positions=bp_positions, sym='', whis=ci)

                # Set the color of the boxes
                plt.setp(bp['boxes'], color=cmap[class_num])
                plt.setp(bp['whiskers'], color=cmap[class_num])
                plt.setp(bp['caps'], color=cmap[class_num])
                for box in bp['boxes']:
                    boxCoords = box.get_xydata()
                    boxPolygon = Polygon(boxCoords, facecolor=cmap[
                                         class_num], alpha=0.7)
                    ax1.add_patch(boxPolygon)

                # If scatter is set to true, plot the scatter for each of the
                # bins for this particular class
                if scatter:
                    for idx, num in enumerate(num_per_label):
                        y_data = np.random.choice(
                            preds[idx, :, class_num].flatten(), size=num)
                        x_data = [bp['whiskers'][idx * 2].get_xdata()[0]] * num
                        jittered_x = _rand_jitter(x_data, bp['boxes'][idx])
                        ax1.scatter(jittered_x, y_data, c=cmap[
                                    class_num], alpha=0.6)

            # Set the bin labels and positions
            xtickNames = plt.setp(
                ax1, xticklabels=[bin_labels.get(label, label) for label in labels])
            plt.setp(ax1, xticks=_get_xtick_loc(num_classes, len(labels)))
            plt.setp(xtickNames, rotation=-45, fontsize=10)
            ax1.set_xlim(0.25, num_boxplots + 0.75)

            # Set up the legend
            patches = [Patch(color=cmap[idx], alpha=0.7, label=class_labels.get(
                idx, idx)) for idx in range(num_classes)]
            plt.legend(handles=patches, loc='best')

        else:
            # Create the boxplots for each label and alter colors
            data = [preds[idx, :] for idx in range(preds.shape[0])]
            bp = plt.boxplot(data, sym='', whis=ci)
            plt.setp(bp['boxes'], color=cmap[0])
            plt.setp(bp['whiskers'], color=cmap[0])
            plt.setp(bp['caps'], color=cmap[0])

            # Fill the boxes with color
            for idx in xrange(len(labels)):
                box = bp['boxes'][idx]
                boxCoords = box.get_xydata()
                boxPolygon = Polygon(boxCoords, facecolor=cmap[0], alpha=0.7)
                ax1.add_patch(boxPolygon)

            # If scatter is set to true, plot the scatter for each of the bins
            if scatter:
                for idx, num in enumerate(num_per_label):
                    y_data = np.random.choice(preds[idx].flatten(), size=num)
                    x_data = [bp['whiskers'][idx * 2].get_xdata()[0]] * num
                    jittered_x = _rand_jitter(x_data, bp['boxes'][idx])
                    ax1.scatter(jittered_x, y_data, c=cmap[1], alpha=0.6)

            # Set the bin labels and positions
            xtickNames = plt.setp(
                ax1, xticklabels=[bin_labels.get(label, label) for label in labels])
            plt.setp(xtickNames, rotation=-45, fontsize=10)
            # ax1.set_xlim(0.25, num_boxplots+0.75)

            if classification:
                # Set up the legend
                patch = Patch(color=cmap[0], alpha=0.7, label=class_labels.get(
                    class_col, class_col))
                plt.legend(handles=[patch], loc='best')

    def _contin_value_plot():
        # We need to set up an array of x_i values to search over
        if num_linspace:
            # Create an interval over +-1 std of the mean of the x column
            mean, std = df[column].mean(), df[column].std()
            lower = np.max([mean - 1.5 * std, df[column].min()])
            upper = np.min([mean + 1.5 * std, df[column].max()])
            x_i = np.linspace(lower, upper, num=num_linspace)
        else:
            # If num_linspace=None, make x_i the unique values
            x_i = np.unique(df[column])

        # For each value in our search space, set the entire column in question to that value and run model.predict or model.predict_proba
        # Average out those predictions and add it to a list of y_hats that we
        # are keeping track of
        preds = []
        for val in x_i:
            dfc[column] = val
            if classification:
                pred = model.predict_proba(dfc)
                preds.append(np.array([boot_sample.mean(axis=0) for boot_sample in (
                    resample(pred) for _ in xrange(1000))]).reshape(-1, pred.shape[1]))
            else:
                pred = model.predict(dfc)
                preds.append(np.array([boot_sample.mean() for boot_sample in (
                    resample(pred) for _ in xrange(1000))]).flatten())

        preds = np.array(preds)

        # Subset down to a particular class if class_col is set
        if classification and class_col:
            preds = preds[:, :, class_col]

        pred_means = preds.mean(axis=1)
        if ci:
            lower_bounds = np.percentile(preds, q=ci[0], axis=1)
            upper_bounds = np.percentile(preds, q=ci[1], axis=1)

        # If preds.shape == 2 then we only need one line and confidence
        # interval
        if len(preds.shape) == 2:
            # Create the fill to indicate the confidence bounds
            if ci:
                ax1.fill_between(x_i, lower_bounds, upper_bounds,
                                 facecolor=cmap[0], alpha=0.2)
            # Plot the predictions
            if classification and class_col:
                ax1.plot(x_i, pred_means, c=cmap[0], linewidth=2, label=class_labels.get(
                    class_col, 'Class {}'.format(class_col)))
                ax1.legend(loc='best')
            else:
                ax1.plot(x_i, pred_means, c=cmap[0], linewidth=2)
        else:
            for col_idx in xrange(preds.shape[2]):
                if ci:
                    ax1.fill_between(x_i, lower_bounds[:, col_idx], upper_bounds[
                                     :, col_idx], facecolor=cmap[col_idx], alpha=0.2)
                ax1.plot(x_i, pred_means[:, col_idx], c=cmap[
                         col_idx], linewidth=2, label=class_labels.get(col_idx, col_idx))
                ax1.legend(loc='best')

        if freq:
            ax2 = ax1.twinx()

            # Set the color of the histogram to be distinct from any of the
            # lines
            if len(preds.shape) == 2:
                freq_color = cmap[1]
            else:
                freq_color = cmap[preds.shape[2] + 1]

            if num_linspace:
                ax2.hist(df.loc[(df[column] >= mean - 1.5 * std) & \
                         (df[column] <= mean + 1.5 * std), column] \
                         .values, facecolor=freq_color, alpha=0.4)
            else:
                ax2.hist(df[column].values, facecolor=freq_color, alpha=0.4)
            ax2.set_ylabel('Frequency')

        # Set xlims to mirror the min and max datapoint
        ax1.set_xlim([x_i.min(), x_i.max()])

    class_labels, cmap = _validate_inputs(class_labels, cmap)

    # Copy our DataFrame so as not to alter the original data
    dfc = df.copy()

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


if __name__ == '__main__':
    ''' Run predictd values plot on regression model '''
    # Read in diamonds dataset
    # See http://docs.ggplot2.org/0.9.3.1/diamonds.html for more info
    df = pd.read_csv('./toy_data/diamonds.csv')

    # Randomly sample 10000 datapoints to make confidence intervals more
    # visible
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
    predicted_value_plot(
        rf, df, 'carat', response_label="Predicted Price in U.S. Dollars")
    plt.savefig('./imgs/regression_predicted_value_plot.png', dpi=300)

    # Run on a discrete column
    bin_labels = {v: k for k, v in cut_map.iteritems()}
    predicted_value_plot(rf, df, 'cut', response_label="Predicted Price in U.S. Dollars",
                         discrete_col=True, bin_labels=bin_labels)
    plt.savefig('./imgs/regression_discrete_predicted_value_plot.png', dpi=300)

    ''' Run predicted values plot on classification model '''
    # Reread in the dataframe
    df = pd.read_csv('./toy_data/diamonds.csv')

    # Subset down to 'Very Good' cut or above
    # df = df.loc[df['cut'].isin(['Ideal', 'Premium', 'Very Good']), :]

    # Map cut to integer label
    cut_map = cut_map = {'Fair': 0, 'Good': 1,
                         'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    class_labels = {v: k for k, v in cut_map.iteritems()}
    df['cut'] = df['cut'].map(lambda x: cut_map[x])

    # Balance our classes so there are 2000 datapoints per cut class
    df = pd.concat([df.loc[df['cut'] == label].sample(n=2000, replace=True,
                                                      random_state=42) for label in df['cut'].unique()], ignore_index=True)

    # Get dummies of some categorical columns
    df = pd.get_dummies(df, columns=['color', 'clarity'], drop_first=True)

    # Make classification model using 'cut' column as target
    y = df.pop('cut').values
    X = df.values

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)

    # Run on a continous column
    predicted_value_plot(
        rf, df, 'price', classification=True, class_labels=class_labels)
    plt.savefig(
        './imgs/classification_continuous_predicted_prob_plot.png', dpi=300)

    # Run on a discrete column
    predicted_value_plot(rf, df, 'color_E', classification=True,
                         discrete_col=True, class_labels=class_labels)
    plt.savefig(
        './imgs/classification_discrete_predicted_prob_plot1.png', dpi=300)

    # Run on a discrete column looking at a particular class
    predicted_value_plot(rf, df, 'color_E', classification=True,
                         discrete_col=True, class_labels=class_labels, class_col=1)
    plt.savefig(
        './imgs/classification_discrete_predicted_prob_plot2.png', dpi=300)
