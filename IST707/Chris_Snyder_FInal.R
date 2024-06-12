#Chris Snyder
#707 final project script (energy)

# pre-processing; creating subsets; plotting figures to answer questions
#performed decision trees, other algorithm attempts...

library(ggplot2)
library(cluster)
library(factoextra)
library(dendextend)
library(dplyr)
library(readr)
library(tidyverse)
library(tidyr)
library(FactoMineR)
library(tree)
library(rpart)
library(rpart.plot)
library(naivebayes)
library(e1071)
library(class)
library(caret)

energydataupdated1 -> energy

#dff = subset(df, select = -c(3, 4, 8, 9, 10, 11, 12, 14, 17, 20, 21, 24, 25, 26, 28) )

#check for nulls + remove
energydataupdated1 -> df
View(energydataupdated)

sum(is.na(df))
#70529

df2 <- df[,-11]
sum(is.na(df2))
#35465

df3 <- df2[,-23]
sum(is.na(df3))
#401

df4<- na.omit(df3)
sum(is.na(df4))
#0

#SUBSET 1 (Aggregated fossil, wind and energy + all other unincluded columns)
FossilEnergy <- df4
na.omit(df4)
sum(is.na(df4))
#View(df4)

#combine columns 3 to 9, 11 & 14, and 20-21
FossilEnergy$FossilEnergySources <- rowSums(FossilEnergy[, 3:9], na.rm = TRUE)
FossilEnergy$HydroGenerationSources <- rowSums(FossilEnergy[, 11:14], na.rm = TRUE)
FossilEnergy$WindGenerationEnergy <- rowSums(FossilEnergy[, c(20, 21)], na.rm = TRUE)

#trying to remove columns
FossilEnergy <- FossilEnergy[, -24]
FossilEnergy <- FossilEnergy[, -22]
FossilEnergy <- FossilEnergy[, -21]
FossilEnergy <- FossilEnergy[, -15]
FossilEnergy <- FossilEnergy[, -14]
FossilEnergy <- FossilEnergy[, -13]
FossilEnergy <- FossilEnergy[, -12]
FossilEnergy <- FossilEnergy[, -11]
FossilEnergy <- FossilEnergy[, -9]
FossilEnergy <- FossilEnergy[, -8]
FossilEnergy <- FossilEnergy[, -7]
FossilEnergy <- FossilEnergy[, -6]
FossilEnergy <- FossilEnergy[, -5]
FossilEnergy <- FossilEnergy[, -4]
FossilEnergy <- FossilEnergy[, -3]

View(FossilEnergy)

#create another aggregated SUBSET, between renewables and nonrenewables:

#subset2
EnergyAG1 <- df4
EnergyAG1$Low_Zero_Emissions <- rowSums(df4[, c(2, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21)])
EnergyAG1$Fossil_High_Emissions <- rowSums(df4[, c(3, 4, 5, 6, 7, 8, 9, 16)])

#removing the original columns that were combined
#subset3
EnergyAG2 <- EnergyAG1[, -c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]

#backkups!!!

Backup1stsubset <- FossilEnergy
Backup2ndsubset <- EnergyAG2

#plots

#1-forecast vs actual load
plot(EnergyAG2$total.load.forecast, EnergyAG2$total.load.actual, main = "Forcasted Total Outputs VS. Actual Total Outputs (Spain, 2015 -2018)", col = "darkblue", xlab = "Total Load Forecast (Megawatts)", ylab = "Total Load Actual (Megawatts)")

#2 - High Emission Outputs by Month (2015-2018)

EnergyAG2$time <- factor(EnergyAG2$time, levels = unique(EnergyAG2$time))

#bar plot
barplot(EnergyAG2$Fossil_High_Emissions, 
        names.arg = EnergyAG2$time,
        main = "High ('Fossil') Emission Energy Source Outputs (Spain, 2015 - 2018)",
        xlab = "Time/Month (2015-1018)",
        ylab = "High Emissions Output (Megawatts)",
        col = "blue",  
        border = "black",
        space = 0.2)  
#trendline

abline(h = mean(EnergyAG2$Fossil_High_Emissions), col = "red", lwd = 2)


#3 - low emission output by month (all years)

EnergyAG2$time <- factor(EnergyAG2$time, levels = unique(EnergyAG2$time))

#bar plot 
barplot(EnergyAG2$Low_Zero_Emissions, 
        names.arg = EnergyAG2$time,
        main = "Low/Zero Emission Energy Source Outputs (Spain, 2015 - 2018)",
        xlab = "Time/Month (2015-2018)",
        ylab = "Low Emissions Output (Megawatts)",
        col = "blue", 
        border = "black",  
        space = 0.2) 
abline(h = mean(EnergyAG2$Low_Zero_Emissions), col = "red", lwd = 2)

#4 - fossil by month combined
custom_order <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#aaggregating the data by month (summing the values for each month)
library(dplyr)
aggregated_data <- EnergyAG2 %>%
  group_by(time) %>%
  summarise(total_energy = sum(Fossil_High_Emissions))

#colorpallete
colors <- rainbow(length(custom_order))

#bar plot with one bar per month
barplot(height = aggregated_data$total_energy[match(custom_order, aggregated_data$time)], 
        names.arg = custom_order, 
        main = "High ('Fossil') Emission Energy Source Outputs by Month (Spain, 2015 - 2018)",
        xlab = "Month",
        ylab = "Summed Outputs From High Emission Energy Sources (Megawatts)",
        col = colors, 
        border = "black",  
        space = 0.2)  
#5 - renewables by month combined

library(dplyr)
aggregated_data2 <- EnergyAG2 %>%
  group_by(time) %>%
  summarise(total_energy = sum(Low_Zero_Emissions))

colors <- rainbow(length(custom_order))

# bar plot 2,one bar per month
barplot(height = aggregated_data2$total_energy[match(custom_order, aggregated_data2$time)], 
        names.arg = custom_order,  # Use the custom order for month labels
        main = "Low/Zero Emission Energy Source Outputs by Month (Spain, 2015-2018)",
        xlab = "Month",
        ylab = "Summed Outputs From Low/Zero Emission Energy Sources (Megawatts)",
        col = colors,  
        border = "black",  
        space = 0.2)  

# 6 - all sources combined

#aggregated data by month(summed)
monthly_aggregated <- aggregate(cbind(Fossil_High_Emissions, Low_Zero_Emissions) ~ time, data = EnergyAG2, sum)

library(reshape2)
monthly_aggregated_melted <- melt(monthly_aggregated, id.vars = "time")

#plotting side by side high and low emissions sources
ggplot(monthly_aggregated_melted, aes(x = time, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  labs(title = "Monthly Comparison of Generated Energy Outputs by Type (Spain, 2015 - 2018)",
       x = "Month",
       y = "Total Output From Sources (Megawatts)") +
  scale_fill_manual(values = c("Fossil_High_Emissions" = "maroon", "Low_Zero_Emissions" = "darkblue")) +
  theme_minimal()

#DECISION TREE

########## Decision Trees
#DT 1
decision_tree <- rpart(Fossil_High_Emissions ~ ., data = EnergyAG1)
print(decision_tree)
library(rpart.plot)
rpart.plot(decision_tree, main = "Decision Tree 1: Predicting Outputs from High Emission Sources")

#DT2
decision_tree1 <- rpart(Low_Zero_Emissions ~ ., data = EnergyAG1)
print(decision_tree1)
library(rpart.plot)
rpart.plot(decision_tree1, main = "Decision Tree 2: Predicting Outputs from Low/Zero Emission Sources")

#RANDOM FOREST ATTEMPT
#Random Forest!

#RF1 (sampledDF)

library(randomForest)
sampledDF$train.label <- as.factor(sampledDF$train.label)
sampledDF1 <- na.omit(sampledDF)

#splitting
set.seed(21)
index_rf1 <- sample(1:nrow(sampledDF), 0.7 * nrow(sampledDF))
train_data_rf1 <- sampledDF1[index_rf1, ]
test_data_rf1 <- sampledDF1[-index_rf1, ]

#mdel create
rf1_model <- randomForest(train.label ~ ., data = train_data_rf1, ntree = 100)
predictions_rf1 <- predict(rf1_model, test_data_rf1)
actual_labels_rf1 <- test_data_rf1$train.label
accuracy_rf1 <- sum(predictions_rf1 == actual_labels_rf1) / length(actual_labels_rf1)

cat("Accuracy:", accuracy_rf1, "\n")
#Accuracy: 0.7394478 -> 73.94%

confusion_rf1 <- confusionMatrix(predictions_rf1, test_data_rf1$train.label)
print(confusion_rf1)


#SUBSET 4 for SVMS (Aggregated fossil, wind and energy + all other unincluded columns)
df5 <- df4
df5$Low_Zero_Emissions <- rowSums(df4[, c(2, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21)])
df5$Fossil_High_Emissions <- rowSums(df4[, c(3, 4, 5, 6, 7, 8, 9, 16)])



#####AR
install.packages("arules")
library(arules)

transactions <- as(df5, "transactions")




###########SVMs

df4 <- data.frame(
  time = c("Jan", "Jan", "Jan", "Jan", "Jan"),
  generation.biomass = c(447, 449, 448, 438, 428),
  # ... (other columns)
  price.actual = c(65.41, 64.92, 64.48, 59.32, 56.04)
)

#target variable
target_variable_name <- "Fossil_High_Emissions"

# training and test
set.seed(21)
train_indices <- sample(1:nrow(EnergyAG2), 0.7 * nrow(EnergyAG2))  # 70% for training
train_data1 <- EnergyAG2[train_indices, ]
test_data1 <- EnergyAG2[-train_indices, ]

# kernel
svm_model <- svm(as.formula(paste(target_variable_name, "~ .")), data = train_data1, kernel = "linear")

#pred
svm_predictions <- predict(svm_model, test_data1)

#eval
accuracy <- sum(svm_predictions == test_data1$target_variable_name) / length(svm_predictions)
print(paste("Accuracy:", accuracy))






