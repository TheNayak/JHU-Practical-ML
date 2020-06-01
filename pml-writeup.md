---
title: "Practical machine learning: predicting classes of excercise."
author: "Rahul Nayak"
date: "01/06/2020"
output:
  html_document:
    df_print: paged
---
********************

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here.](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)  (see the section on the Weight Lifting Exercise Dataset).

## Data loading and processing

The dataset that is provided is split into test and training datasets.
```{r echo=FALSE,message=FALSE,warning=FALSE}
library("ggplot2")
library("lubridate")
library("caret")
library("dplyr")
library("data.table")
library("readr")
```
```{r cache=TRUE,message=FALSE,warning=FALSE}
dat <- read_csv("pml-training.csv")
dat_test<- read_csv("pml-testing.csv")
```
```{r}
head(dat,2)
```
There are many variables with majority NA values and they have to be dropped. The colomn types need to be converted to double and the classe column has to be converted to factor.
```{r}
dat1<-dat[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
dat2<-dat_test[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
```
```{r}
#converting col types
len<-as.numeric(length(dat1[2,]))
for (n in 2:len-1) {
  dat1[[n]]<-as.numeric(dat1[[n]])
}
dat1[[53]]<-as.factor(dat1[[53]])
```
There might be highly correleted variables and they need to be dropped.
```{r}
high.correlation <- findCorrelation(cor(dat1[,-53]), 0.90)
dat1<-dat1[,-high.correlation]
high.correlation1 <- findCorrelation(cor(dat2[,]), 0.90)
dat2<-dat2[,-high.correlation]
```
Next the training data has to be split for training and cross-validation.

The training set needs to be large enough to achieve a relatively high accuracy, and the cross validation set also needs to be large enough to give a good indication of the out of sample error.

The training data set was split up into one portion (70%) for model building, model cohort, and another portion (30%) for cross-validation, cross-validation cohort.
```{r}
#splitting data for cross validation
inTrain <- createDataPartition(dat1$classe, p=.7, list=FALSE)
train<-dat1[inTrain,]
validate<-dat1[-inTrain,]
```

## Model Building

Random forest was chosen as the prediction algorithm used. This model was built on the model cohort and then tested on the cross-validation cohort.

```{r cache=TRUE,results='hide'}
set.seed(46)
model <- train(classe ~., data=train, method="rf", trControl=trainControl(method="cv", verboseIter=TRUE, number=10), ntrees=750)
```


## Confusion Matrix

The confusion matrix allows visualization of the performance of an machine learning algorithm - typically a supervised learning. Each column of the matrix represents the instances in a predicted class, while each row represents the instances in an actual (reference) class.

```{r}
#validation
cross.validation.predict <- predict(model,validate)
#confusion matrix
confusionMatrix <- confusionMatrix(cross.validation.predict, validate$classe)
# Plotting confusion Matrix
cf.table <- as.data.frame(confusionMatrix$table)
ggplot(cf.table, aes(x=Reference, y=Prediction), environment=environment()) +
  geom_tile(fill="white", color="black") +
  geom_text(aes(label=cf.table$Freq)) +
  theme(legend.position="none",  panel.background =element_rect(fill='lightblue') )+
  xlab("Reference") +                    
  ylab("Prediction") 
```


## Accuracy

The random forests model has 99.5% out of sample accuracy, or 0.5% out of sample error.
```{r}
confusionMatrix$overall
```


## Prediction

The model is used to predict classes of the test dataset.

```{r}
testing.predict <- predict(model, dat2)
testing.predict
```

## Summary

The model used was a random forest algorithm using 750 trees. Accuracy obtained was 99.5% with good calculated concordance (kappa = 0.99). The trained algorithm correctly identified 20 out of 20 test cases (the results were considered 100% accurate in the submission part of the project).
