## Practical Machine Learning Course Project
##

library(dplyr, quietly = TRUE, warn.conflicts = FALSE)
library(caret, quietly = TRUE, warn.conflicts = FALSE)

# setwd("./practMLproj")
trainDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainDataFile <- "./pml-training.csv"
# download.file(trainDataUrl, trainDataFile)
tempPmlTrain <- read.csv(trainDataFile)

testDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testDataFile <- "./pml-testing.csv"
# download.file(testDataUrl, testDataFile)
testProbSet <- read.csv(testDataFile)


# Clean up data to remove empty variables
# NOTE: the X variable was removed from the training set because of unnaturally
#   high correlation that was unrealistic.
exerFullData <- subset(tempPmlTrain, select = c("user_name", 
    "new_window", "num_window", "roll_belt",               
    "pitch_belt", "yaw_belt", "total_accel_belt",              
    "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x",            
    "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y",           
    "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm",                 
    "total_accel_arm", "gyros_arm_x",             
    "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y",             
    "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",            
    "roll_dumbbell",           
    "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",             
    "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x",        
    "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y",       
    "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",             
    "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y",         
    "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z",         
    "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "classe"))

# Note that the test problem set differs only in the "classe" variable
# testProbSet <- subset(testProbSetRaw, select = c("user_name",
#     "new_window", "num_window", "roll_belt",               
#     "pitch_belt", "yaw_belt", "total_accel_belt",              
#     "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x",            
#     "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y",           
#     "magnet_belt_z", "roll_arm", "pitch_arm", "yaw_arm",                 
#     "total_accel_arm", "gyros_arm_x",             
#     "gyros_arm_y", "gyros_arm_z", "accel_arm_x", "accel_arm_y",             
#     "accel_arm_z", "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",            
#     "roll_dumbbell",           
#     "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell",             
#     "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x",        
#     "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y",       
#     "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm",             
#     "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y",         
#     "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z",         
#     "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z"))

# Partition dataset for cross validation 
set.seed(050557)
exerTrainIndx <- createDataPartition(exerFullData$classe, p = .75, list = FALSE)
exerTrainBig <- select(exerFullData[exerTrainIndx, ])
exerTest <- select(exerFullData[-exerTrainIndx, ])

# Reduce training set by half
exerTrainIndx2 <- createDataPartition(exerTrainBig$classe, p = .5, list = FALSE)
exerTrain <- select(exerTrainBig[exerTrainIndx2, ])

# Model training using the "randomForest" default in train()
exerMod <- train(classe ~ ., exerTrain)

# Save exerMod from training model
# exerModFile <- paste("exMod", "_rev1", sep = "")
# save(exerMod, file = "./exerModFile")


# Calculate the misclassification rate using missClass modified for factors
missClassFac = function(values,prediction){
    sum(prediction != values)/length(values)
}

trainPred <- predict(exerModTry, newdata = exerTrain)
missClassFac(exerTrain$classe, trainPred)

testPred <- predict(exerModTry, newdata = exerTest)
missClassFac(exerTest$classe, testPred)
confusionMatrix(testPred, exerTest$classe)

# Run model on test problem set
testProbSetPred <- predict(exerModTry, newdata = testProbSet)
testProbSetPred
# With exerModTry using only 1474 observations
# [1] B A B A A E D B A A B C B A E E A B B B
# exerModTry with 1474 gives 
# missClassFac 0.04378819 on exerTestTry
#               0.4608483 on exerTest
#               0.0       on exerTrainTry
#               0.4001902 on exerTrain (which includes exerTrainTry)

# With exterModTry using only 2946 observations
# [1] B A B A A E D B A A B C B A E E A D B B
confusionMatrix(testPred, exerTest$classe)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     A    B    C    D    E
#           A 1391   18    0    0    0
#           B    0  910   18    3    8
#           C    2   17  834   19    3
#           D    1    4    2  779   11
#           E    1    0    1    3  879
# 
# Overall Statistics
# 
# Accuracy : 0.9774          
# 95% CI : (0.9728, 0.9813)
# No Information Rate : 0.2845          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9714          
# Mcnemar's Test P-Value : 3.291e-07       
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9971   0.9589   0.9754   0.9689   0.9756
# Specificity            0.9949   0.9927   0.9899   0.9956   0.9988
# Pos Pred Value         0.9872   0.9691   0.9531   0.9774   0.9943
# Neg Pred Value         0.9989   0.9902   0.9948   0.9939   0.9945
# Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
# Detection Rate         0.2836   0.1856   0.1701   0.1588   0.1792
# Detection Prevalence   0.2873   0.1915   0.1784   0.1625   0.1803
# Balanced Accuracy      0.9960   0.9758   0.9827   0.9823   0.9872

# With exterModTry using only 7360 observations
# [1] B A B A A E D B A A B C B A E E A B B B
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 1391    6    0    0    0
# B    3  938    8    0    2
# C    0    5  846    7    0
# D    0    0    1  796    6
# E    1    0    0    1  893





# 
# Overall Statistics
# 
# Accuracy : 0.9918          
# 95% CI : (0.9889, 0.9942)
# No Information Rate : 0.2845          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9897          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9971   0.9884   0.9895   0.9900   0.9911
# Specificity            0.9983   0.9967   0.9970   0.9983   0.9995
# Pos Pred Value         0.9957   0.9863   0.9860   0.9913   0.9978
# Neg Pred Value         0.9989   0.9972   0.9978   0.9980   0.9980
# Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
# Detection Rate         0.2836   0.1913   0.1725   0.1623   0.1821
# Detection Prevalence   0.2849   0.1939   0.1750   0.1637   0.1825
# Balanced Accuracy      0.9977   0.9926   0.9933   0.9942   0.9953
# 
probAns <- c(1:20, "B", "A", "B", "A", "A", "E", "D", "B", "A", "A", "B", "C", 
             "B", "A", "E", "E", "A", "B", "B", "B")
probAns <- matrix(probAns, ncol = 2, nrow = 20)
probAns
# [,1] [,2]
# [1,] "1"  "B" 
# [2,] "2"  "A" 
# [3,] "3"  "B" 
# [4,] "4"  "A" 
# [5,] "5"  "A" 
# [6,] "6"  "E" 
# [7,] "7"  "D" 
# [8,] "8"  "B" 
# [9,] "9"  "A" 
# [10,] "10" "A" 
# [11,] "11" "B" 
# [12,] "12" "C" 
# [13,] "13" "B" 
# [14,] "14" "A" 
# [15,] "15" "E" 
# [16,] "16" "E" 
# [17,] "17" "A" 
# [18,] "18" "B" 
# [19,] "19" "B" 
# [20,] "20" "B"