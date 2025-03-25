# ==============================================================================
# Copyright (C) 2024 Ozkan Cigdem, Shengjia Chen, Chaojie Zhang, 
# Kyunghyun Cho, Richard Kijowski, Cem M Deniz

# This file is part of 2024_RadiologyAdvances_time2TKR
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
# install.packages("devtools")
# install_github("ksuresh17/autoSurv")
# install.packages("ggplot2")
# install.packages("survival")

library(survival)
library(devtools)
library(ggplot2)
library(autoSurv)
library(caret)
############################# Modified by Ozkan #####################################################

rm(list=ls())
set.seed(1101)

# Read the CSV file.
file_path <- "Data_Preparation/Datasets/SSL_TWIST_Selected_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv" # Update the input data
TKRdata <- read.csv(file_path) 
file_name <- "SSL_TWIST_Selected_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv" 

TKRdata.results <- readRDS("models/RSF_models/SSL_TWIST_Selected_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.RDS") #Update the saved model

TKRdata$statusComp <- ifelse(TKRdata$status == 1, 1, 0)
TKRdata$survTime <- TKRdata$timeVar/365 
TKRdata.dat <- TKRdata[,c(5:ncol(TKRdata))] # The first 4 column are "DataSplits","File_Name","timVar","status"
TKRdata.dat <- TKRdata.dat[complete.cases(TKRdata.dat),] # Get rows without missing data 
##################################################### ################################################
#Standardize covariates
TKRdata.dat_covs <- subset(TKRdata.dat, select = -c(statusComp,survTime))
TKRdata.dat_covs_process <- caret::preProcess(TKRdata.dat_covs, method=c("center", "scale"))
TKRdata.dat_covs <- predict(TKRdata.dat_covs_process, TKRdata.dat_covs)
TKRdata.datProcessed <- cbind(TKRdata[,c("TrainValTest", "survTime","statusComp")], TKRdata.dat_covs)
##################################################### ################################################
TKRdata.dat_train <- TKRdata.datProcessed[TKRdata.datProcessed$TrainValTest == 'train', c(2:ncol(TKRdata.datProcessed))]  # Select rows with 'train' in the 'DataGroup' column
TKRdata.dat_validation <- TKRdata.datProcessed[TKRdata.datProcessed$TrainValTest == 'validation', c(2:ncol(TKRdata.datProcessed))]  # Select rows with 'validation' in the 'DataGroup' column
TKRdata.dat_test <- TKRdata.datProcessed[TKRdata.datProcessed$TrainValTest == 'test',c(2:ncol(TKRdata.datProcessed))]      # Select rows with 'test' in the 'DataGroup' column
###############################################################################################################################################
###############################################################################################################################################
TKRdata.testResults <- predict_autoSurv_TKR(TKRdata.results, 
                                        newdata=TKRdata.dat_test, 
                                        times =  c(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                        timeVar = "survTime", 
                                        statusVar = "statusComp")
###############################################################################################################################################
TKRdata.testResults
mean_auc <- round(100*mean(TKRdata.testResults$metrics$AUC),1)
mean_R2 <- round(mean(TKRdata.testResults$metrics$R2 ),3)
mean_brier <- round(mean(TKRdata.testResults$metrics$Brier),3)
mean_IBS <- round(mean(TKRdata.testResults$metrics$IBS),3)
##############################################################################################################################################
ggplot(data = TKRdata.testResults$auc, aes(y = AUC, x = times, color = model)) +
  geom_point(size = 2) +
  geom_line(size = 0.8) +
  theme_light() +
  theme(text = element_text(size = 15)) +
  xlab("Prediction times") +
  ylab("AUC") +
  guides(color = guide_legend(title = "")) +
  ylim(0, 1) 

ggplot(data = TKRdata.testResults$brier, aes(y = Brier, x = times, color = model)) +
  geom_point(size=2) +
  geom_line(size=0.8) +
  theme_light() + theme(text=element_text(size=15)) + 
  xlab("Prediction times") + ylab("Brier score") +
  guides(color=guide_legend(title=""))
###############################################################################################################################################
###############################################################################################################################################
# Create a list of model names
# model_list <- c("cox", "rsf", "glm")
model_list <- c("rsf")

# Set the number of thresholds and initialize the ACC matrix
num_thresholds <- 10
all_acc_matrix <- list()

num_patients <- nrow(TKRdata.dat_validation)  
all_correctly_predicted_data <-  matrix(0, nrow = num_patients, ncol = 9)

modelno  <- 1
# Loop through each model
for (model_name in model_list) {
  # Initialize the ACC matrix for the current model
  acc_matrix <- matrix(NA, nrow = num_thresholds, ncol = 5)
  correctly_predicted_data <- matrix(0, nrow = num_patients, ncol = 3)
  correctly_predicted_years <- matrix(0, nrow = num_patients, ncol = 3)
  
  # Loop through thresholds from 0.1 to 0.9
  for (i in 1:num_thresholds) {
    threshold <- i * 0.1
    print(threshold)
    
    # Sample predicted probabilities for the current model 
    predicted_probabilities <- TKRdata.results$pred_probabilities[[model_name]]
    
    # Initialize a vector to store the sum of 1s for each patient
    sum_of_1s <- numeric(nrow(predicted_probabilities))
    
    # Loop through each patient
    for (j in 1:nrow(predicted_probabilities)) {
      # Find the time point when the probability crosses the threshold for each patient
      for (k in 1:ncol(predicted_probabilities)) {
        if (predicted_probabilities[j, k] >= threshold) {
          # Increment the sum of 1s
          sum_of_1s[j] <- sum_of_1s[j] + 1
        }
      }
    }
    # Set the tolerance range in years
    tolerance_years <- 0  # Adjust this value as needed
    
    # Loop through tolerance years
    for (tolerance_years in 0:3) {
      # Initialize a counter for true predictions within the tolerance range
      true_predictions <- 0
      
      # Initialize a vector to store 1 or 0 based on condition
      condition_results <- numeric(length(sum_of_1s))
      
      # Set all values in condition_results to zero
      condition_results[] <- 0
      
      # Check for true predictions within the tolerance range
      for (j in 1:length(sum_of_1s)) {
        true_time <- round(TKRdata.dat_validation$survTime[j])
        lower_bound <- true_time - tolerance_years
        upper_bound <- true_time + tolerance_years
        
        if (!is.na(true_time) && !is.na(sum_of_1s[j]) &&
            sum_of_1s[j] >= lower_bound && sum_of_1s[j] <= upper_bound) {
          true_predictions <- true_predictions + 1
          condition_results[j] <- 1
        }
      }
      # Calculate and store the ACC value in the matrix
      ACC <- true_predictions / length(sum_of_1s)
      # print(tolerance_years)
      # print(true_predictions)
      acc_matrix[i, tolerance_years + 1] <- ACC*100
      # Store the results in the correctly_predicted_data matrix
      if (i %in% c(3, 4, 5) && tolerance_years == 1) {
        col_index <- (i - 3) + tolerance_years
        correctly_predicted_data[, col_index] <- condition_results
        correctly_predicted_years[, col_index] <- sum_of_1s
      }
    }
  }
  # Store the ACC matrix for the current model in the list
  all_acc_matrix[[model_name]] <- acc_matrix
  
  # Append the columns from correctly_predicted_data to all_correctly_predicted_data
  all_correctly_predicted_data[, (modelno):(modelno + 2)] <- correctly_predicted_data
  modelno <- modelno + 3
}
########################################################################################################################
# To save the predicted years and predictions results(0 is wrong prediction 1 is true prediction) in a csv file
combined_correctly_predicted_data_years = data.frame(Labels = round(TKRdata.dat_validation$survTime), Years = correctly_predicted_data, score = correctly_predicted_years )

# Print the list of ACC matrices for all models
print(all_acc_matrix)
###############################################################################################################################################
###############################################################################################################################################
# Find the row number with the highest value in the second column
# bestAcc_row <- which.max(all_acc_matrix[[model_name]][, 2])
bestAcc_threshold <- 0.4
print(bestAcc_threshold)
###############################################################################################################################################
###############################################################################################################################################
num_patientsTest <- nrow(TKRdata.dat_test)   
all_acc_matrixTest <- list()

# Loop through each model
for (model_nameTest in model_list) {
  # Initialize the ACC matrix for the current model
  acc_matrixTest <- matrix(NA, nrow = 1, ncol = 4) # In acc_matrixTest we see the accuracy of test model for original year, -+1year, -+2years, -+3years
  correctly_predicted_dataTest <- matrix(0, nrow = num_patientsTest, ncol = 1)
  correctly_predicted_yearsTest <- matrix(0, nrow = num_patientsTest, ncol = 1)
  
  # Sample predicted probabilities for the current model 
  predicted_probabilitiesTest <- TKRdata.testResults$pred_probabilities[[model_nameTest]]
  
  # Initialize a vector to store the sum of 1s for each patient
  sum_of_1sTest <- numeric(nrow(predicted_probabilitiesTest))
  
  # Loop through each patient
  for (j in 1:nrow(predicted_probabilitiesTest)) {
    # Find the time point when the probability crosses the bestAcc_threshold for each patient
    for (k in 1:ncol(predicted_probabilitiesTest)) {
      if (predicted_probabilitiesTest[j, k] >= bestAcc_threshold) {  # If the probability is higher or equal then the threshold value, then add 1 
        # Increment the sum of 1s
        sum_of_1sTest[j] <- sum_of_1sTest[j] + 1                         # Sum no of 1s to find the prediction of time to TKR surgery
      }
    }
  }
  # Set the tolerance range in years
  tolerance_yearsTest <- 0  # tolerance_yearsTest is for -+1year, -+2years, -+3years
  
  # Loop through tolerance years
  for (tolerance_yearsTest in 0:3) {
    # Initialize a counter for true predictions within the tolerance range
    true_predictionsTest <- 0
    # Initialize a vector to store 1 or 0 based on condition
    condition_resultsTest <- numeric(length(sum_of_1sTest))
    
    # Check for true predictions within the tolerance range
    for (j in 1:length(sum_of_1sTest)) {
      true_timeTest <- round(TKRdata.dat_test$survTime[j])
      lower_boundTest <- true_timeTest - tolerance_yearsTest
      upper_boundTest <- true_timeTest + tolerance_yearsTest
      
      if (!is.na(true_timeTest) && !is.na(sum_of_1sTest[j]) &&
          sum_of_1sTest[j] >= lower_boundTest && sum_of_1sTest[j] <= upper_boundTest) {
        true_predictionsTest <- true_predictionsTest + 1
        condition_resultsTest[j] <- 1
      }
    }
    # Calculate and store the ACCTest value in the matrix
    ACCTest <- true_predictionsTest / length(sum_of_1sTest)
    acc_matrixTest[1, tolerance_yearsTest + 1] <- round(ACCTest*100,1)
    # Store the results in the correctly_predicted_data matrix
    if (tolerance_yearsTest == 1) {
      correctly_predicted_dataTest[, tolerance_yearsTest] <- condition_resultsTest 
      correctly_predicted_yearsTest[, tolerance_yearsTest] <- sum_of_1sTest 
    }
  }
  # Store the ACCTest matrix for the current model in the list
  all_acc_matrixTest[[model_nameTest]] <- acc_matrixTest
}

###############################################################################################################################################
# Define a function to calculate Concordance Index
calculate_cindex <- function(predictions, true_times) {
  # Fit a Cox proportional hazards model
  cox_model <- coxph(Surv(true_times) ~ predictions)
  
  # Calculate C-index using survival::concordance()
  c_index <- survival::concordance(cox_model)
  
  return(round((c_index$concordance)*100,1))
}

# Calculate Concordance Index (C-index) for the current model
cindex <- calculate_cindex(sum_of_1sTest, round(TKRdata.dat_test$survTime))

auc_values <- TKRdata.testResults$metrics$AUC

# Calculate the mean of the AUC values
mean_auc <- mean(auc_values)

# Calculate the standard deviation of AUC values
sd_auc <- sd(auc_values)

# Number of AUC values (n = 10)
n <- length(auc_values)

# Calculate the standard error
se_auc <- sd_auc / sqrt(n)

# t-critical value for 95% confidence interval (n-1 degrees of freedom)
t_critical <- qt(0.975, df = n - 1)

# Calculate the 95% confidence interval
ci_lower <- mean_auc - t_critical * se_auc
ci_upper <- mean_auc + t_critical * se_auc

# Display the results
mean_auc_rounded <- round(100 * mean_auc, 1)
ci_lower_rounded <- round(100 * ci_lower, 1)
ci_upper_rounded <- round(100 * ci_upper, 1)

# Print mean values
cat("Mean AUC:", mean_auc, "\n")
cat("Mean R2:", mean_R2, "\n")
cat("Mean Brier:", mean_brier, "\n")
cat("Mean IBS:", mean_IBS, "\n")
print(all_acc_matrixTest)
print(cindex)
cat("Mean AUC:", mean_auc_rounded, "\n")
cat("95% Confidence Interval: [", ci_lower_rounded, ",", ci_upper_rounded, "]\n")
###############################################################################################################################################
###############################################################################################################################################
