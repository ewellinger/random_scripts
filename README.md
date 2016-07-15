# Random Scripts
This is simply a repository of useful scripts I have written.


## Predicted Value Plot
Script for performing a predicted value plot for a particular feature given a model, dataframe, and column in question.  You can use this script to get a sense for how changing the value of a particular feature will influence your predicted value in the case of regression, or predicted probability in the case of a binary classification.

The following graphic is the resulting plot using the standard Boston dataset included with the scikit-learn library looking at the 'CRIM' feature (per capita crime rate by town).

![Predicted Values Plot of Column CRIM Using a Random Forest Model](./imgs/regression_predicted_value_plot.png)
