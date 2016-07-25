# Random Scripts
This is simply a repository of useful scripts I have written.


## Predicted Value Plot
Script for performing a predicted value plot for a particular feature given a model, dataframe, and column in question.  You can use this script to get a sense for how changing the value of a particular feature will influence your predicted value in the case of regression, or predicted probability in the case of a binary classification.

The following graphic is the resulting plot using the standard Boston dataset included with the scikit-learn library looking at the 'CRIM' feature (per capita crime rate by town).

![Predicted Values Plot of Column CRIM Using a Random Forest Model](./imgs/regression_predicted_value_plot.png)

The following graphic is the resulting plot when running the `predicted_value_plot` function on a column containing discrete values.  The entire column will be reset to the value of each discrete value and predictions will be made.  The box plot of predictions is then generated from the mean predictions of 1000 bootstrapped samples of those predictions.  There is also an optional parameter for superimposing a jittered scatter plot of bootstrapped means over the box plot.

![Predicted Values Plot of Discrete Column CHAS Using a Random Forest Model](./imgs/regression_discrete_predicted_value_plot.png)


**NOTE**: In the case of a classification model, this script currently only works for a binary classification model.  A future improvement would be to extend the function's ability to consider a columns affect on each classes' predicted probability.
