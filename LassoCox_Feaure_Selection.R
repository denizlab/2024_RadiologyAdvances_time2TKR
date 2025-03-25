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
# Install the 'glmnet' package if not already installed
if (!require(glmnet)) {
  install.packages("glmnet")
  library(glmnet)
}
if (!require(survival)) {
  install.packages("survival")
  library(survival)
}
if (!require(caret)) {
  install.packages("caret")
  library(caret)
}
########################################################################################

library(survival)
library(devtools)
library(ggplot2)
library(autoSurv)
library(glmnet)
#########################################################################################
rm(list=ls())
set.seed(1101)
file_path <- "Data_Preparation/Datasets/SSL_TWIST_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv"
TKRdata <- read.csv(file_path)  

file_name <- "SSL_Selected_TWIST_DESS_TSE_Xray_MOAKS_Quantitative_Radiographic_Imputed.csv" 

subject_group <- sapply(strsplit(as.character(TKRdata$File_Name), "_"), function(x) x[1])

# Preprocess the data
TKRdata$statusComp <- ifelse(TKRdata$status == 1, 1, 0)
TKRdata$survTime <- TKRdata$timeVar / 365
TKRdata.dat <- TKRdata[, c(5:ncol(TKRdata))]

# Standardize covariates
TKRdata.dat_covs <- subset(TKRdata.dat, select = -c(statusComp, survTime))
TKRdata.dat_covs_process <- caret::preProcess(TKRdata.dat_covs, method = c("center", "scale"))
TKRdata.dat_covs <- predict(TKRdata.dat_covs_process, TKRdata.dat_covs)
TKRdata.datProcessed <- cbind(TKRdata[, c("TrainValTest", "survTime", "statusComp")], TKRdata.dat_covs)

# Separate datasets
TKRdata.dat_train <- TKRdata.datProcessed[TKRdata.datProcessed$TrainValTest == 'train', c(2:ncol(TKRdata.datProcessed))]

# Define the formula for survival analysis
surv_formula <- Surv(survTime, statusComp) ~ .

# Extract features and survival data for training
features_train <- subset(TKRdata.dat_train, select = -c(statusComp, survTime))
surv_data_train <- Surv(TKRdata.dat_train$survTime, TKRdata.dat_train$statusComp)

# Ensure features_train is a matrix
features_train <- as.matrix(features_train)

# Ensure the column names are not empty
colnames(features_train) <- make.names(colnames(features_train))

# Fit the LASSO-Cox model
lasso_cox_model <- cv.glmnet(x = features_train,y = surv_data_train,family = "cox",alpha = 1,folds = subject_group)  # Use the custom grouping variable

# Print the optimal lambda value selected by cross-validation
print(paste("Optimal Lambda:", lasso_cox_model$lambda.min))

# Extract the coefficients of the selected features
lasso_coefficients <- coef(lasso_cox_model, s = "lambda.min")
selected_features <- rownames(lasso_coefficients)[lasso_coefficients[, 1] != 0]

# Print or use the selected features
cat("Selected Features:/n")
print(selected_features)

# Create a data frame with the selected features
selected_data <- as.data.frame(features_train[, selected_features])
selected_data$survival <- TKRdata.dat_train$survTime
selected_data$statusComp <- TKRdata.dat_train$statusComp

lasso_cox_model <- coxph(Surv(survival, statusComp) ~ ., data = selected_data)
# Print the summary of the LASSO Cox model
summary(lasso_cox_model)


# Get the absolute values of the coefficients
absolute_coefficients <- abs(coef(lasso_cox_model))

# Create a dataframe to store the variable names and their absolute coefficient values
importance_df <- data.frame(
  Variable = names(absolute_coefficients),
  Absolute_Coefficient = absolute_coefficients
)

# Sort the dataframe based on the absolute coefficient values to rank the variables
ranked_importance <- importance_df[order(importance_df$Absolute_Coefficient, decreasing = TRUE), ]

ranked_importance$Relative_Importance <- ranked_importance$Absolute_Coefficient / min(ranked_importance$Relative_Importance)

# Print or view the ranked importance dataframe
print(ranked_importance)
write.csv(ranked_importance, file = "Data_Preparation/Datasets/Variable_Importance.csv")

# Extract hazard ratios and 95% confidence intervals
hr_ci <- exp(coef(lasso_cox_model))
conf_int <- confint(lasso_cox_model)

# Create a data frame to store the results
results_df <- data.frame(
  Feature = names(hr_ci),
  Hazard_Ratio = hr_ci,
  Lower_CI = conf_int[, 1],
  Upper_CI = conf_int[, 2]
)

# Print the results
print(results_df)
write.csv(results_df, file = "Data_Preparation/Datasets/Variable_Importance_HazardRatios.csv")

# Create a function to filter and select features for a given dataset
filter_and_select_features <- function(dataset) {
  dataset[, c("TrainValTest","File_Name","survTime", "statusComp", selected_features)]
}
# Apply the function to train, validation, and test datasets
selected_combined_data <- filter_and_select_features(TKRdata)

selected_combined_data$survTime <- selected_combined_data$survTime * 365

# Rename the 'statusComp' column to 'status'
names(selected_combined_data)[names(selected_combined_data) == "statusComp"] <- "status"
names(selected_combined_data)[names(selected_combined_data) == "survTime"] <- "timeVar"

file_save_dir <- "Data_Preparation/Datasets"
combined_file_path <- file.path(file_save_dir, file_name)
# Save the data to a CSV file
write.csv(selected_combined_data, file = combined_file_path, row.names = FALSE)
