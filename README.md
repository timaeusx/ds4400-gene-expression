# Classifying Acute Forms of Leukemia using Gene Expression Data
Final project files and data repository for DS4400 Machine Learning and Data Mining at Northeastern University

Amanda Bell, Timothy Wang, Jasmine Wong

Project code can be found in the [notebook](https://github.com/timaeusx/ds4400-gene-expression/blob/main/DS_4400_Gene_Expression_Project_Code.ipynb) or on [Google Colab](https://colab.research.google.com/drive/163GDGjxep-fIH8uGLzC3-ZxdNnKgV24I?usp=sharing).

## Abstract

Leukemia is a family of blood cancers that is predicted to affect almost 60,000 Americans in 2023. Different forms of leukemia require differnt treatments, so it is essential that physicians have as many tools as possible to help them classify differnt forms of leukemia. Using a dataset published by Golub et al. [1], this project aims to investigate and build a machine learning model that can effectively help physicians differentiate two acute forms of leukemia: Acute Myeloid Leukemia (AML) and Acute Lymphocytic Leukemia (ALL), which are severe conditions with low five-year survival rates. In search of the best model, logistic regressors, random forest classifiers, k-nearest neighbors classifiers, and a fully connected neural network were tested. Additional attempts to improve the performance of the models included hyperparameter tuning, feature selection, cross validation and dimension reduction. Ultimately, out of the nine machine learning models built, the model with the highest accuracy was the logistic regressor without principal component analysis, which performed with approximately 97% accuracy. In a testing set of 34 patients, the one error was incorrectly classifying ALL as AML.

## Introduction

Leukemia is a family of blood cancers, the exact cause of which remains unknown. The American Cancer Society estimates that in 2023, almost 60,000 new cases of leukemia and 23,710 deaths from leukemia will occur [2]. There are four major types of leukemia:
- Acute Myeloid Leukemia (AML)
- Acute Lymphocytic Leukemia (ALL)
- Chronic Myeloid Leukemia (CML)
- Chronic Lymphocitic Leukemia (CLL)

as well as several other minor types.

It is difficult to classify the different types of leukemia by symptoms alone, but some predictive factors like genetics, family history, and environmental exposure have been observed in correlation. With recent improvements in sequencing technologies and advancements in computing power, gene expression analysis is rapidly becoming a feasible option for diagnosing and treating many types of cancers, including leukemia.

This project aims to investigate and build a machine learning model that can effectively help physicians accurately diagnose different forms of leukemia. A high accuracy model can play a significant role in helping to deliver optimal treatment plans to combat these deadly cancers. Along with traditional data analysis methods such as logistic regressors, random forests, and nearest-neighbors classifiers, neural networks represent interesting areas of exploration for solving the problem of classifying and differentiating between the various types of leukemia. We apply the aforementioned methods to a dataset published by Golub et al. [1], connecting gene analysis metrics with diagnosed instances of AML and ALL. Our approaches attempt to use a wide array of gene expression features to predict the occurence of either AML or ALL in each case.

As a caveat, we note that the dataset used is limited in size, as although there are upwards of 7,000 features, there are only 72 total data points. This is most likely because is often difficult to obtain patient data due to a combination of privacy, disease progression, and other factors. Another limitation is that there is a label imbalance in the data; that is, the dataset contains 42 data points associated with ALL, but only 25 data points associated with AML.

## Dataset and experimental design

The dataset used was published by Golub et al. [1] and contains the genetic data of 72 patients. 

In general, the dataset has the following features:
- Gene Description 
- Gene Accession Number
- Relative gene expression values - scaled for comparability
- Gene call: a decision made on whether a gene is present
  - Absent (A): The gene is absent.
  - Present (P): The gene is present.
  - Marginal (M): Too close to call; the gene may be absent or present.
  
We predict a binary target variable, indicating either AML or ALL.

The following models were applied to the dataset:
- Principal Component Analysis/Regression
- Logistic Regression (with and without PCA dimensionality reduction)
- Random Forest Classification (with and without dimensionality reduction)
- K-Nearest Neighbors Classification
- Fully Connected Neural Network
  - 7129 inputs
  - 128 hidden units
  - 2 outputs

Grid seach cross validation was used to determine the ideal parameters for PCA, random forest, and KNN models.

## Results
_Describe the results from your experiments._
- Main results: Describe the main experimental results you have; this is where you highlight the most interesting findings.
- Supplementary results: Describe the parameter choices you have made while running the experiments. This part goes into justifying those choices.

### Principal Component Analysis/Regression
Using cross validation, we determined that a linear regression model run on the dataset reduced to 35 principal components achieved the lowest mean squared error (MSE) score. This model was validated on the test set and achieved an R^2 score of about 0.72 and MSE score of 0.06. We applied this dimensionality reduction to the rest of the models to see if there was any improvement, but in most cases saw a reduction in accuracy.

### Logistic Regression (with and without PCA dimensionality reduction)
Logistic regression was one of the best performing models even without changing any parameters or reducing features with a test accuracy of 97% and only one misclassification where an AML patient was predicted to have ALL. We were curious if we could improve performance even more by reducing the number of dimensions with PCA considering our dataset had over 7000 features, however we saw a drop in performance to 88% accuracy and an increase to 4 false negatives. 

### Random Forest Classification (with and without dimensionality reduction, reduced number of features)
Random forest classifier started off initially with a train error of 0.0 and a test error of 0.24 and a train F1 of 1.0 and a test F1 of 0.83 potentially indicating some overfitting of the model to the dataset. Hyperparameter tuning for n_estimators, min_samples_leaf, and min_samples_split proved this to be true with an increase in test error to 0.26 and a decrease in F1 to 0.82. 9 patients were classified with AML although they were ALL patients. Poor classification performance was thought to be a result of too many features for the random forest classifier to handle so we tried 3 different approaches. First we took the 100 most important features, found the best selection of those features (70-80 total) going in order of importance, and trained a model to get a test error of 0.12 and test F1 of 0.91 with only 4 misclassifications of ALL as AML. Then we tried dimension reduction using PCA but achieved only similar results as we had originally. Lastly, we were curious about using only genes indicated in ALL and AML so we selected a subset of about 70 of those genes and used them to train another RFC but similarly achieved only similar results as we had originally.

### K-Nearest Neighbors Classification
The K-Nearest Neighbors Classifier performed relatively well on the testing data, with a prediction accuracy of 88.24%. In order to build the KNN classifier, we used GridSearch to find the best parameter for the number of neighbors. A n_neighbors that is too large can lead to underfitting, while a n_neighbors that is too small can lead to overfitting. Through GridSearch, it was extracted that the optimal number of neighbors was 4, and this was used to build the final KNN classifier.

### Fully Connected Neural Network
The neural network used had 7129 input units, 128 hidden units, and 2 output units. The network achieved a minimum cross entropy loss of about 0.08, and accuracy score of 71.05%. 

## Discussion
_Discuss the results obtained above. If your results are very good, see if you could compare them with some existing approaches that you could find online. If your results are not as good as you had hoped for, make a good-faith diagnosis about what the problem is._

After experiementing with a number of machine learning models, we were able to build a Logistic Regression model that performed with 97% accuracy. With only one incorrect classification, this is a very powerful model that can provide value to medical professionals who might need to differentiate between AML vs. ALL.

## Conclusion
_In several sentences, summarize what you have done in this project._

This project explored the classification problem of two Leukemia types (AML and ALL) through building nine machine learning models. The machine learning models built include Principal Component Analysis and Regression, Logistic Regression without PCA, Logistic Regression with PCA, a Random Forest Classifier without PCA, a Random Forest Classifier with Reduced Number of Features using Feature Importances, a Random Forest Classifier with PCA, a Random Classifier with Selection of Genes indicated by AML and ALL, K-Nearest Neighbors, and a Fully Connected Neural Network. The best performing model was the Logistic Regressor without PCA dimension reduction, which performed with a 97% accuracy. Since the Logistic Regressor was one of the first models our team built, we worked on attempting to improve other models such as the Random Forest Classifier through adjusting strategies such as testing different features by filtering with feature importance, princial component analysis, and only training on a subset of 70 genes associated with ALL and AML. This project successfully resulted in a high-accuracy classifier that was able to distinguish AML and ALL, as well as a meaningful learning experience about machine learning through the iterative process of building models, testing them, and improving them through trying different strategies.

## References
[1] T. R. Golub et al., Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring. Science 286, 531-537 (1999). DOI:10.1126/science.286.5439.531

[2] American Cancer Society. Cancer Facts & Figures 2023. Atlanta, Ga: American Cancer Society; 2023.

Kaggle repository with data: https://www.kaggle.com/datasets/crawford/gene-expression

Genes for ALL: https://www.mycancergenome.org/content/disease/acute-lymphoblastic-leukemia/#:~:text=Overview&text=Acute%20lymphoblastic%20leukemias%20most%20frequently,%2C%20and%20USP7%20%5B2%5D.&text=WT1fs%2C%20NOTCH1%20Mutation%2C%20NOTCH1%20Missense,acute%20lymphoblastic%20leukemia%20%5B2%5D.

Gene for AML: Lagunas-Rangel, F. A., Chávez-Valencia, V., Gómez-Guijosa, M. Á., & Cortes-Penagos, C. (2017). Acute Myeloid Leukemia-Genetic Alterations and Their Clinical Prognosis. International journal of hematology-oncology and stem cell research, 11(4), 328–339.
