# Classifying Acute Forms of Leukemia using Gene Expression Data
Final project files and data repository for DS4400 Machine Learning and Data Mining at Northeastern University
Amanda Bell, Timothy Wang, Jasmine Wong

## Abstract
_In several sentences, summarize your project (e.g., think of a Tweet)._

Leukemia is a family of blood cancers that is predicted to affect almost 60,000 Americans in 2023. Different forms of leukemia require differnt treatments, so it is essential that physicians have as many tools as possible to help them classify differnt forms of leukemia. Using a dataset published by Golub et al. [1], this project aims to investigate and build a machine learning model that can effectively help physicians differentiate two acute forms of leukemia: Acute Myeloid Leukemia(AML) and Acute Lymphocytic Leukemia(ALL), which are severe conditions with low five-year survival rates. In search of the best model, logistic regressors, random forest classifiers, k-nearest neighbors classifiers, and a fully connected neural network were tested. Additional attempts to improve the performance of the models included hyperparameter tuning, cross validation and dimension reduction. Ultimately, the model with the highest accuracy was the logistic regressor without principal component analysis, which performed with approximately 97% accuracy. In a testing set of 34 patients, the one error was incorrectly classifying ALL as AML.

## Introduction
- What is the problem? For example, what are you trying to solve? Describe the motivation.
- Why is this problem interesting? Is this problem helping us solve a bigger task in some way? Where would we find use cases for this problem?
- What is the approach you propose to tackle the problem? What approaches make sense for this problem? Would they work well or not? Feel free to speculate here based on what we taught in class.
- Why is the approach a good approach compared with other competing methods? For example, did you find any reference for solving this problem previously? If there are, how does your approach differ from theirs?
- What are the key components of my approach and results? Also, include any specific limitations.

Leukemia is a family of blood cancers, the exact cause of which remains unknown. The American Cancer Society estimates that in 2023, almost 60,000 new cases of leukemia and 23,710 deaths from leukemia will occur. There are four major types of leukemia:
- Acute Myeloid Leukemia (AML)
- Acute Lymphocytic Leukemia (ALL)
- Chronic Myeloid Leukemia (CML)
- Chronic Lymphocitic Leukemia (CLL)

as well as several other minor types.

It is difficult to classify the different types of leukemia by symptoms alone, but some predictive factors like genetics, family history, and environmental exposure have been observed in correlation. With recent improvements in sequencing technologies and advancements in computing power, gene expression analysis is rapidly becoming a feasible option for diagnosing and treating many types of cancers, including leukemia.

This project aims to investigate and build a machine learning model that can effectively help physicians accurately diagnose different forms of leukemia. A high accuracy model can play a significant role in helping to deliver optimal treatment plans to combat these deadly cancers. Along with traditional data analysis methods such as logistic regressors, random forests, and nearest-neighbors classifiers, neural networks represent interesting areas of exploration for solving the problem of classifying and differentiating between the various types of leukemia. We apply the aforementioned methods to a dataset published by Golub et al. [1], connecting gene analysis metrics with diagnosed instances of AML and ALL. Our approaches attempt to use a wide array of gene expression features to predict the occurence of either AML or ALL in each case.

As a caveat, we note that the dataset used is limited in size, as although there are upwards of 7,000 features, there are only 72 total data points. This is most likely because is often difficult to obtain patient data due to a combination of privacy, disease progression, and other factors. Another limitation is that there is a label imbalance in the data; that is, the dataset contains 42 data points associated with ALL, but only 25 data points associated with AML.

## Dataset and experimental design
- Describe the dataset, including its basic statistics.
- Describe the experimental setup, including what models you are going to run, what parameters you plan to use, and what computing environment you will execute on.
- Describe the problem setup (e.g., for neural networks, describe the network structure that you are going to use in the experiments).

In general, the dataset has the following features:
- Gene Description 
- Gene Accession Number
- Numbers for each patient - values for gene expression
- Call for each gene for a patient
  - Absent (A)
  - Present (P)
  - Marginal (M)
  
We predict a binary target variable, indicating either AML or ALL.

The following models were applied to the dataset:
- Principal Component Analysis/Regression
- Logistic Regression (with and without PCA dimensionality reduction)
- Random Forest Classification (with and without dimensionality reduction)
- K-Nearest Neighbors Classification
- Fully Connected Neural Network

## Results
_Describe the results from your experiments._
- Main results: Describe the main experimental results you have; this is where you highlight the most interesting findings.
- Supplementary results: Describe the parameter choices you have made while running the experiments. This part goes into justifying those choices.

### Principal Component Analysis/Regression

### Logistic Regression (with and without PCA dimensionality reduction)

### Random Forest Classification (with and without dimensionality reduction)

### K-Nearest Neighbors Classification

### Fully Connected Neural Network

## Discussion
_Discuss the results obtained above. If your results are very good, see if you could compare them with some existing approaches that you could find online. If your results are not as good as you had hoped for, make a good-faith diagnosis about what the problem is._

## Conclusion
_In several sentences, summarize what you have done in this project._

## References
[1] T. R. Golub et al., Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring. Science 286, 531-537 (1999). DOI:10.1126/science.286.5439.531
<br>Kaggle repository with data: https://www.kaggle.com/datasets/crawford/gene-expression
