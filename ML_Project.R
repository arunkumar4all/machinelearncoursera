library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(rmarkdown)
library(knitr)

training_raw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing_raw = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings=c("NA","#DIV/0!",""))


training <- training_raw
testing <- testing_raw
set.seed(100)

# remove near zero variance
var_nz <- nearZeroVar(training)
training <- training[,-var_nz]
testing <- testing[,-var_nz]


training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]

training   <-training[,-c(1:5)]
testing <-testing[,-c(1:5)]

# create training data partition for testing
traindata <- createDataPartition(y=training$classe, p=0.80, list=FALSE)
training_1 <- training[traindata, ] 
training_2 <- training[-traindata, ]


ds_model <- rpart(classe~.,data = training_1, method="class")
ds_predict <- predict(ds_model,training_2,type="class")
fancyRpartPlot(ds_model, palettes=c("Greys", "Oranges"))
confusionMatrix(ds_predict, training_2$classe)

rf_model <- randomForest(classe~.,data = training_1)
rf_predict <- predict(rf_model, training_2,type="class")
confusionMatrix(rf_predict, training_2$classe)


# suppressWarnings(
#   gbm_model <- train(classe~., method="gbm", data=training_1, 
#                  verbose="FALSE", trControl=trainControl(method="cv")))
# print(gbm_model)


rf_predict_cv <-predict(rf_model,testing, type= "class")
rf_predict_cv




  

