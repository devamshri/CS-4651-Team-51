# Welcome!

## People:
#### Shikhar Ahuja, Abhinav Govindaraju, Akhil Kammila, Devam Shrivastava, Ryan Zhang

## Introduction
The finance industry presents a constant challenge to data-driven investors. An investor who finds meaningful patterns in stock price datasets can stand to make significant amounts of money. For this reason, there has been tremendous research in analyzing stock price datasets with various techniques. Employing ML is no exception.

There are countless ML strategies that are commonly used to analyze stocks. Some include NLP for sentiment analysis, deep learning for pattern recognition, and transfer learning for directional stock predictions. Our project will focus on two specific techniques. The first is Principal Component Analysis, which can help us reduce the number of variables that we are analyzing. From there, we will perform correlation analysis to determine which stocks are highly correlated. This will allow us to develop information for pairs trading strategies.

## Background
Vast amounts of research has been conducted into both PCA and correlation analysis on stocks. M Ghorbani, for instance, conducted research employing PCA to predict future stock prices of 150 different companies in 2020. He compared PCA to both Gauss-Bayes, which is more computationally expensive, and moving average, which is much simpler. He found that PCA worked better than both of the other methods. PCA has also been shown to improve performances of SVM and linear regression models. This literature informed our own decision to employ PCA mixed with correlation analysis.

## Problem
Our problem is to identify correlations between financial variables in order to develop mean reversion models. We found that traditional methods of analyzing stock time series data often encounter difficulties in handling the high dimensionality inherent in such datasets. In this context, Principal Component Analysis (PCA) emerges as a promising tool to discern and prioritize the underlying factors influencing stock movements. Our dataset will be Yahoo finance data on the top 100 Average Price Dollar stocks. We will use data spanning from the past 1.5 years to the past 0.5 years. We are planning to use daily open, close, high and low prices, along with the stock volume, dividends and stock splits.

## Methods: 
We intend to employ Primary Component Analysis as a method of feature extraction (reducing the number of stocks that we are analyzing). From there, we intend to test correlations on mean reversions of stock prices and also perform linear regressions of stock prices. Our models will be implemented using scikit-learn and numpy/pandas. Additionally, we will test our algorithms on both daily returns and log daily returns to see which type of data can yield more information. 

## Results and Significance:
There are several aspects of the project that can be quite significant. Successful PCA analysis can successfully extract the most important features from the dataset. Additionally, developing correlational models for mean reversion will help financial models accurately gain an “edge” on the market. In the context of trading, this could mean a highly successful trading strategy, which would generate positive gains. If our work leads to profitable trading strategies, we hope to backtest and employ them in real financial markets.


## Sources:

Anass Nahil, Abdelouahid Lyhyaoui. Short-term stock price forecasting using kernel principal component analysis and support vector machines: the case of Casablanca stock exchange, Procedia Computer Science, Volume 127, 2018, pp. 161-169, doi: https://doi.org/10.1016/j.procs.2018.01.111.

Ghorbani M, Chong EKP. Stock price prediction using principal components. PLoS One. 2020 Mar 20;15(3):e0230124. doi: 10.1371/journal.pone.0230124. PMID: 32196528; PMCID: PMC7083277.

M. Waqar, H. Dawood, P. Guo, M. B. Shahnawaz and M. A. Ghazanfar, "Prediction of Stock Market by Principal Component Analysis," 2017 13th International Conference on Computational Intelligence and Security (CIS), Hong Kong, China, 2017, pp. 599-602, doi: 10.1109/CIS.2017.00139.

## Extra
Another study conducted in 2017 describes a method using PCA to reduce noise amongst financial data. This highlights attributes that are most relevant to the prediction problem. The resulting data is applied onto a linear regression model for prediction. The data used in the paper is from various stock exchanges including NYSE, London Stock Exchange, and Karachi Stock Exchange. It demonstrates how PCA can be beneficial if the principal components chosen are well correlated to the desired predictions.
