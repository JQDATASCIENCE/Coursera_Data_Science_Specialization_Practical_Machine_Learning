---
title: "Practical Machine Learning Wk 4 Proj"
author: "SNG JUN QIANG"
date: "9 March 2019"
output:
    html_document:
            keep_md: true
---

<style>
body {
text-align: justify}
</style>



### 1. Background

Devices such as Jawbone Up, Nike FuelBand, and Fitbit is collecting large amount of data about personal activity relatively inexpensively. These devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

### 2. Objective

Using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

### 3. Data Source

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) 

The test data are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) 

Citation: [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). 

### 4. LIbraries

```r
library(RColorBrewer)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-14
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

### 5. Loading Data

Loading driectly from web and downloaded as data frame.

```r
TrainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestURL  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
Train <- read.csv(url(TrainURL))
Test <- read.csv(url(TestURL))
```

### 6. Data Exploration and Processing

From exploration below, total variables = 160 and totala observation = Train(19,622) + Test(20) = 19,642.
Train contains 1,287,472 missing values which is about 41%. Test contains 2,000 missing values which is about 63%. There are total 5 levels in the label variable classe and there are no NA in variable classe. From the plot, classe A has the most frequency followerd by classe E.


```r
dim(Train)
```

```
## [1] 19622   160
```

```r
dim(Test)
```

```
## [1]  20 160
```

```r
sum(is.na(Train))
```

```
## [1] 1287472
```

```r
sum(is.na(Test))
```

```
## [1] 2000
```

```r
table(Train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
str(Train$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
sum(is.na(Train$classe))
```

```
## [1] 0
```

```r
plot(Train$classe, col="#FF3333")
```

![](Practical_Machine_Learning_Wk_4_Proj_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

Removing unecessary columns for both Train and Test. Replacing blank cell with NA. Converting all predictors to numeric.


```r
Sub_Train <- Train[,-c(1:7)]
Sub_Train[Sub_Train == ""] <- NA
Sub_Train[, 1:152] <- sapply(Sub_Train[, 1:152], as.numeric)
Sub_Train[is.na(Sub_Train)] <- 0
sum(is.na(Sub_Train))
```

```
## [1] 0
```

```r
Sub_Test <- Test[,-c(1:7)]
Sub_Test[Sub_Test == ""] <- NA
Sub_Test[, 1:152] <- sapply(Sub_Test[, 1:152], as.numeric)
Sub_Test[is.na(Sub_Test)] <- 0
sum(is.na(Sub_Test))
```

```
## [1] 0
```

### 7. Cross Validation Set


```r
set.seed(123)
Index <- createDataPartition(Sub_Train$classe, p=0.75)[[1]]
Index_Sub_Train <- Sub_Train[Index,]
Index_Sub_Train_CROSSV <- Sub_Train[-Index,]
dim(Index_Sub_Train)
```

```
## [1] 14718   153
```

```r
dim(Index_Sub_Train_CROSSV)
```

```
## [1] 4904  153
```

### 8. Decision Tree Model

Predicting using decision tree model shows only 49.7% accuracy which is not a desired prediction.


```r
DT <- train(classe ~ ., data = Index_Sub_Train, method="rpart")
DT_predict <- predict(DT, Index_Sub_Train_CROSSV)
confusionMatrix(DT_predict, Index_Sub_Train_CROSSV$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1262  378  418  356  144
##          B   20  312   26  141  101
##          C  107  259  411  307  248
##          D    0    0    0    0    0
##          E    6    0    0    0  408
## 
## Overall Statistics
##                                           
##                Accuracy : 0.488           
##                  95% CI : (0.4739, 0.5021)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3307          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9047  0.32877  0.48070   0.0000  0.45283
## Specificity            0.6307  0.92718  0.77254   1.0000  0.99850
## Pos Pred Value         0.4934  0.52000  0.30856      NaN  0.98551
## Neg Pred Value         0.9433  0.85200  0.87570   0.8361  0.89020
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2573  0.06362  0.08381   0.0000  0.08320
## Detection Prevalence   0.5216  0.12235  0.27162   0.0000  0.08442
## Balanced Accuracy      0.7677  0.62797  0.62662   0.5000  0.72567
```

```r
rpart.plot(DT$finalModel, roundint=FALSE)
```

![](Practical_Machine_Learning_Wk_4_Proj_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

### 9. Random Forest Model

Predicting using random forest model shows only 99.4% accuracy which is a good prediction.


```r
RF <- train(classe ~ ., data = Index_Sub_Train, method = "rf", ntree = 100)
RF_predict <- predict(RF, Index_Sub_Train_CROSSV)
confusionMatrix(RF_predict, Index_Sub_Train_CROSSV$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    5    0    0    0
##          B    1  943    4    0    0
##          C    0    1  848    7    0
##          D    0    0    3  796    2
##          E    0    0    0    1  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9927, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9937   0.9918   0.9900   0.9978
## Specificity            0.9986   0.9987   0.9980   0.9988   0.9998
## Pos Pred Value         0.9964   0.9947   0.9907   0.9938   0.9989
## Neg Pred Value         0.9997   0.9985   0.9983   0.9981   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1923   0.1729   0.1623   0.1833
## Detection Prevalence   0.2853   0.1933   0.1746   0.1633   0.1835
## Balanced Accuracy      0.9989   0.9962   0.9949   0.9944   0.9988
```

### 10. Conclusion

Random Forest Model is performing better giving a 99.4% accuracy as compared to Decision Tree Model. Hence, Random Forest Model will be use for the prediction.

### 11. Applying Random Forest Model to Final Prediction


```r
Final_Predict <- predict(RF,Sub_Test)
Final_Predict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
