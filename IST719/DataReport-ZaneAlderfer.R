hearts <- read.csv("/Users/zanealderfer/Downloads/heart_statlog_cleveland_hungary_final.csv", 
                    sep = ",", header=TRUE)

str(hearts)
dim(hearts)

library(ggplot2)

ggplot(hearts, aes(x = age)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 20) +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency") +
  theme_minimal() +
  theme(plot.caption = element_text(size = 8, hjust = 1), legend.position = "none") +
  annotate("text", x = Inf, y = Inf, hjust = 1, vjust = 1, label = "Source: Heart Disease Data")

ggplot(hearts, aes(y = cholesterol)) +
  geom_boxplot(fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Cholesterol Levels", x = "", y = "Cholesterol Level (mg/dl)") +
  theme_minimal() +
  theme(plot.caption = element_text(size = 8, hjust = 1), legend.position = "none") +
  annotate("text", x = Inf, y = Inf, hjust = 1, vjust = 1, label = "Source: Heart Disease Data")

ggplot(hearts, aes(x = factor(chest.pain.type))) +
  geom_bar(fill = "salmon", color = "black") +
  labs(title = "Frequency of Chest Pain Types", x = "Chest Pain Type", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.caption = element_text(size = 8, hjust = 1), legend.position = "none") +
  annotate("text", x = Inf, y = Inf, hjust = 1, vjust = 1, label = "Source: Heart Disease Data")

ggplot(hearts, aes(x = age, y = max.heart.rate, color = factor(target))) +
  geom_point() +
  labs(title = "Age vs. Max Heart Rate (Colored by Target)", x = "Age", y = "Max Heart Rate", color = "Target") +
  theme_minimal() +
  theme(plot.caption = element_text(size = 8, hjust = 1), legend.position = "bottom") +
  annotate("text", x = Inf, y = Inf, hjust = 1, vjust = 1, label = "Source: Heart Disease Data")
