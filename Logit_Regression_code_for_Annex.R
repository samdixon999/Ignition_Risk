
# install and load required packages

install.packages("corrplot")
install.packages("caret")
install.packages("regclass")
install.packages("raster")
install.packages("ROCR")
install.packages("pROC")
install.packages("rgdal")
install.packages("e1071")

library(raster)
library(dplyr)
library(ROCR)
library(pROC)
library(corrplot)
library(caret)
library(regclass)
library(rgdal)
library(e1071)

############################

# set working directory
setwd("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Rebuilding_Logistic")

# create file list to create raster stack
fs <- list.files(path = "U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Rebuilding_Logistic\\Raster_inputs", pattern = "tif$", full.names = TRUE)

# create raster stack using file list
rasstack <- stack(fs)

###########################

# Create non ignition sample points
# load non ignition area
# Non ignition area was created in ArcMap. It is the study area minus ignition loacation plus a 200m buffer.
non_ignit_area <- readOGR(dsn="U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test", layer="Non_ignit_area_200m")

# set seed
# this was changed for each sample run
set.seed(1)

# create random sample points in non ignition area
NonIgnitSample <- spsample(non_ignit_area,n=222,"random")

# write random sample coordinates to csv
write.csv(NonIgnitSample@coords, file = "U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\nonignitpoints1.csv", row.names = FALSE)

#####################

# load ignition and non ignition points
# Ignition points derive from the MFFP wildfire database
Ignition <- read.csv("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Rebuilding_Logistic\\Sample_points\\Ignition_points_09to18.csv")
Non_Ignition <- read.csv("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\nonignitpoints1.csv")

# set csv files to spatial points data frame
coordinates(Ignition) <- ~ Point_X + Point_Y
coordinates(Non_Ignition) <- ~ x + y
Non_Ignition <- as(Non_Ignition,"SpatialPointsDataFrame")

# extract raster values from sample points
Ignition_values <- extract(rasstack,Ignition)
Non_Ignition_values <- extract(rasstack,Non_Ignition)

# Combine raster values with points
Ignition_Comb <- cbind(Ignition,Ignition_values)
Non_Ignition_Comb <- cbind(Non_Ignition@coords,Non_Ignition_values)

# write combined files to csv for future reference
write.csv(Ignition_Comb, file = "U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Ignition_rast_values.csv", row.names = FALSE)
write.csv(Non_Ignition_Comb, file = "U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Non_Ignition_rast_values_200m_v1.csv", row.names = FALSE)

#########################

# Add ignition and non ignition raster data to environment ready for combination 
Ignition_csv <- read.csv("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Ignition_rast_values.csv")
Non_Ignition_csv <- read.csv("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Non_Ignition_rast_values_200m_v1.csv")

# Add column to give ID for whether sample point is an ignition source or not
Ignition_csv['Ignition'] = 1
Non_Ignition_csv['Ignition'] = 0

# Combine the two datasets to give the final data to build the model
Build_data <- rbind(Ignition_csv, Non_Ignition_csv)

# write the build data to csv for future reference
write.csv(Build_data, file = "U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Model_build_data_09-18_200m_v7.csv", row.names = FALSE)

# if already built input data can load from file
# Build_data <- read.csv("U:\\GIS\\Projects\\Fire_Risk_Map\\Analysis\\Testing_Logistic\\Non_Ignition_test\\Model_build_data_09-18_200m_v1.csv")

# view correlation matrix between all predictor variables
correls <- cor(Build_data[,2:10])
corrplot(correls, method = "circle")

# extract training and testing data
set.seed(2018)
n <- nrow(Build_data)
shuffled_df <- Build_data[sample(n), ]
train_indices <- 1:round(0.6 * n)
train <- shuffled_df[train_indices, ]
test_indices <- (round(0.6 * n) + 1 ):n
test <- shuffled_df[test_indices, ]

####################################

# build model using all variables ready for the backwards selection
allvar <- glm(Ignition ~ dist_CarPark + dist_IMD1to3 + dist_LayBy + dist_Major + dist_Minor + dist_PROW + dist_PWHiPop + dist_Wayline + LCM2015_clipped, data = train, family = binomial())

# summary stats of the model
summary(allvar)

# Run the backwards stepwise regression
# using k=log(n) which uses the BIC function for selecting the model. 
# using the BIC functiuon as it seems to be quite stringent on what it includes
backwards <- step(allvar, k=log(n))
summary(backwards)

# calculate and plot ROC AUC on test data
p <- predict(backwardsselect, newdata = test, type = "response")
pr <- prediction(p, test$Ignition)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# build confusion matrix
confusionMatrix(table(predict(backwardsselect, newdata = test, type = "response") >= 0.5,
                      test$Ignition == 1))
