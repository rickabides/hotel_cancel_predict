# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(summarytools)
library(DataExplorer)

# Read the CSV file (assuming you're using the full dataset now)
data <- read.csv("HotelReservations_trim_v2.csv")

# Prepare data
colSums(is.na(data))
hotel_data <- data %>% select(-BookingID,-ArrivalDate,-ArrivalYear,-ArrivalMonth,-NoOfWeekendNights,-NoOfWeekNights)
summary(hotel_data)

create_report(hotel_data)

# Data preprocessing
# Define categorical variables for one-hot encoding
categorical_vars <- c("TypeOfMealPlan", "RoomTypeReserved", "MarketSegmentType", "BookingStatus")

# Create dummy variables
dummy_model <- dummyVars(~ ., data = hotel_data, fullRank = TRUE)
hotel_data_encoded <- data.frame(predict(dummy_model, newdata = hotel_data))

# Convert binary variables to numeric
hotel_data_encoded$RequiredCarParkingSpace <- as.numeric(as.character(hotel_data$RequiredCarParkingSpace))
hotel_data_encoded$RepeatedGuest <- as.numeric(as.character(hotel_data$RepeatedGuest))

# Create a new variable for total nights
hotel_data_encoded$TotalNights <- hotel_data$NoOfWeekendNights + hotel_data$NoOfWeekNights

# Remove unnecessary columns
#hotel_data_encoded <- hotel_data_encoded %>% 
 # select()

# Split the data into training and testing sets
set.seed(42)
train_index <- createDataPartition(hotel_data$BookingStatus, p = 0.8, list = FALSE)
train_data <- hotel_data_encoded[train_index, ]
test_data <- hotel_data_encoded[-train_index, ]

# Fit the logistic regression model
logistic_model <- glm(BookingStatusNot_Canceled ~ ., data = train_data, family = "binomial")

# Print model summary
summary(logistic_model)

# Make predictions on the test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to class predictions
predicted_classes <- ifelse(predictions > 0.5, "Not_Canceled", "Canceled")

# Create confusion matrix
conf_matrix <- table(Predicted = predicted_classes, Actual = test_data$BookingStatusNot_Canceled)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy, 4)))

# Calculate AUC-ROC
roc_obj <- roc(test_data$BookingStatusNot_Canceled, predictions)
auc_value <- auc(roc_obj)
print(paste("AUC-ROC:", round(auc_value, 4)))

# Plot ROC curve
plot(roc_obj, main = "ROC Curve")

# Extract values from confusion matrix
TP <- conf_matrix[1, 1]
TN <- conf_matrix[2, 2]
FP <- conf_matrix[1, 2]
FN <- conf_matrix[2, 1]

# Accuracy
accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Display the results
cat("Confusion Matrix:\n")
print(conf_matrix)

cat("\nPerformance Metrics:\n")
cat("Accuracy: ", round(accuracy, 4), "\n")
cat("Precision: ", round(precision, 4), "\n")
cat("Recall: ", round(recall, 4), "\n")
cat("F1 Score: ", round(f1_score, 4), "\n")

# Feature importance (based on absolute z-values)
model_summary <- summary(logistic_model)
feature_importance <- data.frame(
  Feature = rownames(model_summary$coefficients)[-1],
  Importance = abs(model_summary$coefficients[-1, "z value"])
)

# Sort feature importance and print
feature_importance <- feature_importance[order(-feature_importance$Importance), ]
print(feature_importance)

# Plot feature importance
ggplot(head(feature_importance, 10), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Feature", y = "Importance (|z-value|)")

# Odds ratios
odds_ratios <- exp(coef(logistic_model))
odds_ratios_df <- data.frame(
  Feature = names(odds_ratios),
  OddsRatio = odds_ratios
)
options(scipen = 999)
odds_ratios_df <- odds_ratios_df[order(-odds_ratios_df$OddsRatio), ]
print(head(odds_ratios_df, 10))


# Load necessary libraries
library(car) # For VIF calculation

alias_info <- alias(logistic_model)
print(alias_info)
# Calculate Variance Inflation Factor (VIF) to check for multicollinearity
vif_values <- vif(logistic_model)
print(vif_values)

# Extract p-values from the model summary
p_values <- summary(logistic_model)$coefficients[, "Pr(>|z|)"]

# Identify insignificant features (p-value > 0.05)
insignificant_features <- names(p_values)[p_values > 0.001]

# Print insignificant features for verification
print(insignificant_features)

# Create a new formula excluding insignificant features
# Ensure to remove the intercept if it's aliased
significant_formula <- as.formula(paste("BookingStatusNot_Canceled ~",
                                        paste(names(coef(logistic_model))[-1][!(names(coef(logistic_model))[-1] %in% insignificant_features)], collapse = " + ")))

# Refit the logistic regression model without insignificant features
logistic_model_refit <- glm(significant_formula, data = train_data, family = "binomial")

# Print the summary of the refitted model
summary(logistic_model_refit)

# Make predictions with the refitted model
new_predictions <- predict(logistic_model_refit, newdata = test_data, type = "response")
new_predicted_classes <- ifelse(new_predictions > 0.5, 1, 0)

# Create confusion matrix for the refitted model
new_conf_matrix <- table(Predicted = new_predicted_classes, Actual = test_data$BookingStatusNot_Canceled)
print(new_conf_matrix)

# Recalculate performance metrics for the refitted model
new_TP <- new_conf_matrix[1, 1]
new_TN <- new_conf_matrix[2, 2]
new_FP <- new_conf_matrix[1, 2]
new_FN <- new_conf_matrix[2, 1]

new_accuracy <- (new_TP + new_TN) / (new_TP + new_TN + new_FP + new_FN)
new_precision <- new_TP / (new_TP + new_FP)
new_recall <- new_TP / (new_TP + new_FN)
new_f1_score <- 2 * (new_precision * new_recall) / (new_precision + new_recall)

# Display the results for the refitted model
cat("\nRefitted Model Performance Metrics:\n")
cat("Accuracy: ", round(new_accuracy, 4), "\n")
cat("Precision: ", round(new_precision, 4), "\n")
cat("Recall: ", round(new_recall, 4), "\n")
cat("F1 Score: ", round(new_f1_score, 4), "\n")
d

