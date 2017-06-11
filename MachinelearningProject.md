# Coursera: Practical Machine Learning Prediction Assignment
Arun R [GitHub](https://github.com/arunkumar4all/machinelearncoursera)  

> **Background**
> Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


> **Data **
> The training data for this project are available here: 
> https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

> The test data are available here: 
> https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

> The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

> **What you should submit**
> The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

> 1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
> 2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

> **Reproducibility **
> Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 


# Prepare the datasets

Setting up all libraries.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(rmarkdown)
library(knitr)
```

Read the training and testing data into a data frame. For reproducibility we set seed to be 100.


```r
training_raw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing_raw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

training <- training_raw
testing <- testing_raw
set.seed(100)
```

# Data preprocessing

1. Removing all nor-zero variance variables would reduce number of covariates. We do this using nearZeroVar function.
2. Remove all the rows that would render zero's which wouldnt add any meaningful values
3. After performing step 1 and step 2, we still have columns that arent useful - we remove columns 1 to 5 to have only useful variables


```r
var_nz <- nearZeroVar(training)
training <- training[,-var_nz]
testing <- testing[,-var_nz]

training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]

training   <-training[,-c(1:5)]
testing <-testing[,-c(1:5)]
```

## Partitioning

Creating 80% and 20% data partitioning would give us a good sub samples to train and cross validate before applying the model on testing


```r
traindata <- createDataPartition(y=training$classe, p=0.80, list=FALSE)
training_1 <- training[traindata, ] 
training_2 <- training[-traindata, ]
```

# Model Creation

## First model: Using Decision Tree



```r
ds_model <- rpart(classe~.,data = training_1, method="class")
ds_predict <- predict(ds_model,training_2,type="class")
fancyRpartPlot(ds_model, palettes=c("Greys", "Oranges"))
```

![](MachinelearningProject_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
confusionMatrix(ds_predict, training_2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 980 117  20  33  30
##          B  40 471  49  64  81
##          C   8  55 561  94  49
##          D  52  91  40 424  73
##          E  36  25  14  28 488
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7453          
##                  95% CI : (0.7314, 0.7589)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6774          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8781   0.6206   0.8202   0.6594   0.6768
## Specificity            0.9287   0.9260   0.9364   0.9220   0.9678
## Pos Pred Value         0.8305   0.6681   0.7314   0.6235   0.8257
## Neg Pred Value         0.9504   0.9105   0.9610   0.9325   0.9301
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2498   0.1201   0.1430   0.1081   0.1244
## Detection Prevalence   0.3008   0.1797   0.1955   0.1733   0.1507
## Balanced Accuracy      0.9034   0.7733   0.8783   0.7907   0.8223
```

## Second model: Using Random Forest


```r
rf_model <- randomForest(classe~.,data = training_1)
rf_predict <- predict(rf_model, training_2,type="class")
confusionMatrix(rf_predict, training_2$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  757    1    0    0
##          C    0    1  683    1    0
##          D    0    0    0  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   0.9984   1.0000
## Specificity            0.9996   0.9997   0.9994   1.0000   1.0000
## Pos Pred Value         0.9991   0.9987   0.9971   1.0000   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1637   0.1838
## Detection Prevalence   0.2847   0.1932   0.1746   0.1637   0.1838
## Balanced Accuracy      0.9998   0.9985   0.9990   0.9992   1.0000
```


# Conclusion:

Random forest method seems to be accurate among the two models we tested with 99.9% accuracy while decision tree has shown a relatively poor accuracy of 75% .

## Applying final model on test data

Finally we apply the random forecast model to the test data to predict outcomes


```r
rf_predict_cv <-predict(rf_model,testing, type= "class")
rf_predict_cv
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




---
references:
- id: fenner2012a
  URL: 'https://tgmstat.wordpress.com/2014/03/06/near-zero-variance-predictors'
  DOI: 10.1038/nmat3283
---







