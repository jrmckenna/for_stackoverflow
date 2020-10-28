library(tidyverse)
library(fastDummies)
library(keras)
library(caret)
library(tensorflow)
library(dplyr)
library(ggplot2)
library(reticulate)
library(devtools)


df <-read.csv("sampledata.csv", header= TRUE)
df<-df[1:1000 , 2:7]
head(df)
barplot(prop.table(table(df$Found)),
        col = rainbow(3),
        ylim = c(0, 0.8),
        ylab = 'Proportion',
        xlab = 'Found',
        cex.names = 1.5)
df[1:6] <- lapply(df[1:6], as.numeric)
df<- as.matrix(df)
dimnames(df) <- NULL 


# # Normalize 
df[,1:5] <- normalize(df[,1:5])
df[ ,6] <- as.numeric(df[ ,6])
str(df)

# Partition
set.seed(1234)
ind <- sample(2, nrow(df), replace = T, prob=c(.7, .3))
training <- df[ind==1, 1:5]
test <- df[ind==2, 1:5]
trainingtarget <- df[ind==1, 6]
testtarget <- df[ind==2, 6]

# One hot encoding 
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels[1:10,])


# Create sequential model and add layers 
model <- keras_model_sequential()
model %>%  layer_dense(units = 8, activation = 'relu', input_shape = ncol(training)) %>%   
  layer_dense(units = 3, activation = 'softmax') 

# Model summary
summary(model)


 model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


 # Fit 
 model_one <- model %>%   
   fit(training, 
       trainLabels, 
       epochs = 100,
       batch_size = 32, 
       validation_split = 0.2)

plot(model_one)

summary(model)
model %>% evaluate(X_test, y_test)

model_one_df <- as.data.frame(model_one_df)
str(model_one_df)




































# 
# as_metrics_df = function(history) {
#   
#   # create metrics data frame
#   df <- as.data.frame(history$metrics)
#   
#   # pad to epochs if necessary
#   pad <- history$params$epochs - nrow(df)
#   pad_data <- list()
#   for (metric in history$params$metrics)
#     pad_data[[metric]] <- rep_len(NA, pad)
#   df <- rbind(df, pad_data)
#   
#   # return df
#   df
# }












plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="orange", type="l")
lines(history$metrics$val_loss, col="skyblue")
legend("topright", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))

plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="orange", type="l")
lines(history$metrics$val_acc, col="skyblue")
legend("topleft", c("Training","Testing"), col=c("orange", "skyblue"), lty=c(1,1))

predictions <- model %>% predict_classes(X_test)
table(factor(predictions, levels=min(final.test$Kyphosis_present):max(final.test$Kyphosis_present)),factor(final.test$Kyphosis_present, levels=min(final.test$Kyphosis_present):max(final.test$Kyphosis_present)))

# conda install h5py

save_model_hdf5(model, "rkerasmodel.h5")

model <- load_model_hdf5("rkerasmodel.h5")


















