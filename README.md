# PCA_predictive_power_study
An empirical study to find out if retaining explained variance in principal component analysis(PCA) helps retain predictive power.

In part 1, PCA_predictive_power_question.ipynb, a fake dataset is created to illustrate that PCs with lower explained variance may have better predictive power than the top ones.

Part 2, PCA_emipiricals.ipynb, aims at finding out, empirically, whether higher explained variance do come with better predictive power in real life cases.
10 datasets are collected and minimally cleaned. Each has a binary target variable. 
PCAs are conducted to get two groups of PCs, the first 5 by the explained variance, and the 6th to 10th.
An ensemble model with 10 fold cross validation is trained on each PC group and their AUC scores are compared.

Available on medium: [Part 1](https://medium.com/digital-alchemist/principal-component-analysis-does-higher-explained-variance-mean-more-predictive-power-f59606ed1e7), [Part 2](https://medium.com/digital-alchemist/principal-component-analysis-does-higher-explained-variance-mean-more-predictive-power-24d888478807)

