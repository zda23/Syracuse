data <- read.csv("/Users/zanealderfer/Downloads/heart_statlog_cleveland_hungary_final.csv", 
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


ggplot(data, aes(x=age, fill=factor(sex))) + 
  geom_histogram(binwidth=1, position="dodge") +
  labs(title="Age Distribution by Sex", x="Age", y="Count", fill="Sex")

data$age_group <- cut(data$age, breaks=c(20, 30, 40, 50, 60, 70, 80), right=FALSE)
ggplot(data, aes(x=age_group, y=cholesterol)) + 
  geom_boxplot() +
  labs(title="Cholesterol Levels by Age Group", x="Age Group", y="Cholesterol")

ggplot(data, aes(x=age, y=resting.bp.s)) + 
  geom_point(alpha=0.5) +
  geom_smooth(method="lm", color="red") +
  labs(title="Resting Blood Pressure vs. Age", x="Age", y="Resting Blood Pressure")

ggplot(data, aes(x=age, y=max.heart.rate, color=factor(sex))) + 
  geom_point(alpha=0.5) +
  labs(title="Max Heart Rate by Age and Sex", x="Age", y="Max Heart Rate", color="Sex")

ggplot(data, aes(x=factor(chest.pain.type), fill=factor(target))) + 
  geom_bar(position="dodge") +
  labs(title="Chest Pain Type Distribution", x="Chest Pain Type", y="Count", fill="Heart Disease")

#install.packages("reshape2")
#install.packages("ggcorrplot")
library(reshape2)
library(ggcorrplot)
cor_matrix <- cor(data[,sapply(data, is.numeric)])
ggcorrplot(cor_matrix, lab = TRUE)

ggplot(data, aes(x=factor(resting.ecg), fill=factor(target))) + 
  geom_bar(position="dodge") +
  labs(title="Distribution of Resting ECG Results", x="Resting ECG", y="Count", fill="Heart Disease")

ggplot(data, aes(x=oldpeak, y=factor(ST.slope), color=factor(target))) + 
  geom_jitter(alpha=0.5) +
  labs(title="Oldpeak vs. ST Slope", x="Oldpeak", y="ST Slope", color="Heart Disease")

ggplot(data, aes(x=age_group, fill=factor(target))) + 
  geom_bar(position="dodge") +
  labs(title="Heart Disease Presence by Age Group", x="Age Group", y="Count", fill="Heart Disease")

ggplot(data, aes(x=age, fill=factor(exercise.angina))) + 
  geom_histogram(binwidth=1, position="dodge") +
  facet_wrap(~sex) +
  labs(title="Exercise-Induced Angina by Sex and Age", x="Age", y="Count", fill="Exercise Angina")

ggplot(data, aes(x=cholesterol, y=resting.bp.s, color=factor(chest.pain.type))) + 
  geom_point(alpha=0.5) +
  labs(title="Cholesterol vs. Resting Blood Pressure Colored by Chest Pain Type", x="Cholesterol", y="Resting Blood Pressure", color="Chest Pain Type")

ggplot(data, aes(x=age, y=max.heart.rate, color=factor(exercise.angina))) + 
  geom_point(alpha=0.5) +
  labs(title="Max Heart Rate vs. Age Colored by Exercise Angina", x="Age", y="Max Heart Rate", color="Exercise Angina")

ggplot(data, aes(x=age, y=oldpeak, color=factor(ST.slope))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~sex) +
  labs(title="Age vs. Oldpeak by ST Slope and Sex", x="Age", y="Oldpeak", color="ST Slope")

ggplot(data, aes(x=age, y=cholesterol, color=factor(sex))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~target) +
  labs(title="Cholesterol vs. Age by Sex and Heart Disease Presence", x="Age", y="Cholesterol", color="Sex")

ggplot(data, aes(x=factor(resting.ecg), y=max.heart.rate, fill=factor(chest.pain.type))) + 
  geom_boxplot() +
  facet_wrap(~target) +
  labs(title="Resting ECG vs. Max Heart Rate by Chest Pain Type and Heart Disease Presence", x="Resting ECG", y="Max Heart Rate", fill="Chest Pain Type")

ggplot(data, aes(x=age, y=resting.bp.s, color=factor(chest.pain.type))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~sex) +
  labs(title="Resting Blood Pressure vs. Age by Chest Pain Type and Sex", x="Age", y="Resting Blood Pressure", color="Chest Pain Type")

ggplot(data, aes(x=cholesterol, fill=factor(sex))) + 
  geom_density(alpha=0.5) +
  facet_wrap(~age_group) +
  labs(title="Cholesterol Distribution by Age Group and Sex", x="Cholesterol", y="Density", fill="Sex")

ggplot(data, aes(x=oldpeak, fill=factor(target))) + 
  geom_density(alpha=0.5) +
  facet_wrap(~ST.slope) +
  labs(title="Oldpeak Distribution by ST Slope and Heart Disease Presence", x="Oldpeak", y="Density", fill="Heart Disease")

ggplot(data, aes(x=cholesterol, y=max.heart.rate, color=factor(exercise.angina))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~target) +
  labs(title="Max Heart Rate vs. Cholesterol by Exercise Angina and Heart Disease Presence", x="Cholesterol", y="Max Heart Rate", color="Exercise Angina")

ggplot(data, aes(x=cholesterol, y=resting.bp.s, color=factor(sex))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~target) +
  labs(title="Resting Blood Pressure vs. Cholesterol by Sex and Heart Disease Presence", x="Cholesterol", y="Resting Blood Pressure", color="Sex")

#install.packages("GGally")
library(GGally)
ggpairs(data, columns = 1:11, aes(color=factor(target), alpha=0.5)) +
  labs(title="Pairwise Plot of All Numerical Variables Colored by Heart Disease Presence")

ggplot(data, aes(x=age, y=cholesterol, color=factor(target))) + 
  geom_point(alpha=0.5) +
  facet_grid(chest.pain.type ~ exercise.angina) +
  labs(title="Age vs. Cholesterol with Facets for Chest Pain Type and Exercise Angina", x="Age", y="Cholesterol", color="Heart Disease")

#install.packages("plotly")
library(plotly)
plot_ly(data, x = ~age, y = ~cholesterol, z = ~resting.bp.s, color = ~factor(target), colors = c('#BF382A', '#0C4B8E')) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Age'),
                      yaxis = list(title = 'Cholesterol'),
                      zaxis = list(title = 'Resting Blood Pressure')),
         title = "3D Scatter Plot of Age, Cholesterol, and Resting Blood Pressure")

install.packages("MASS")
library(MASS)
library(GGally)
ggparcoord(data, columns = c(1, 5, 6, 8, 11), groupColumn = 12, scale="std", alphaLines=0.5) +
  labs(title="Parallel Coordinates Plot of Key Variables Colored by Heart Disease Presence", x="Variables", y="Standardized Values", color="Heart Disease")

install.packages("fmsb")
library(fmsb)
data_radar <- data[, c("age", "cholesterol", "resting.bp.s", "max.heart.rate", "oldpeak", "target")]
data_radar_summary <- aggregate(. ~ target, data_radar, mean)
data_radar_summary <- rbind(rep(200,6) , rep(0,6) , data_radar_summary)
radarchart(data_radar_summary, axistype=1,
           pcol=c("#1f77b4","#ff7f0e"), pfcol=c("#1f77b480","#ff7f0e80") , plwd=2, 
           cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0,200,50), cglwd=0.8,
           vlcex=0.8,
           title="Radar Chart of Average Values for Different Variables Grouped by Heart Disease Presence")
legend(x=1.1, y=1, legend = c("No Heart Disease","Heart Disease"), bty = "n", pch=20, col=c("#1f77b4","#ff7f0e"), text.col = "black", cex=1.2, pt.cex=3)

library(ggcorrplot)
cor_matrix <- cor(data[,sapply(data, is.numeric)])
ggcorrplot(cor_matrix, method = "circle", lab = TRUE, lab_size = 3, colors = c("blue", "white", "red"), 
           title = "Heatmap of Correlations with Annotations", ggtheme = theme_minimal())

ggplot(data, aes(x=age, y=cholesterol, color=factor(target))) + 
  geom_point(alpha=0.5) +
  facet_grid(resting.ecg ~ chest.pain.type) +
  labs(title="Facet Grid of Age vs. Cholesterol by Resting ECG and Chest Pain Type", x="Age", y="Cholesterol", color="Heart Disease")

data$age_group <- cut(data$age, breaks=seq(20, 80, by=5))
data$cholesterol_group <- cut(data$cholesterol, breaks=seq(100, 400, by=20))
heatmap_data <- table(data$age_group, data$cholesterol_group, data$target)
heatmap_data <- as.data.frame(as.table(heatmap_data))
ggplot(heatmap_data, aes(Var1, Var2, fill=Freq)) + 
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title="Heatmap of Heart Disease Presence by Age and Cholesterol", x="Age Group", y="Cholesterol Group", fill="Frequency") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(data, aes(x=cholesterol, y=max.heart.rate, size=age, color=factor(target))) + 
  geom_point(alpha=0.5) +
  scale_size(range = c(1, 10)) +
  labs(title="Bubble Chart of Cholesterol vs. Max Heart Rate by Age Group and Heart Disease Presence", x="Cholesterol", y="Max Heart Rate", color="Heart Disease", size="Age")

ggplot(data, aes(x=age, y=cholesterol, color=factor(target))) + 
  geom_point(alpha=0.5) +
  labs(title="Age vs. Cholesterol Colored by Heart Disease Presence", x="Age", y="Cholesterol", color="Heart Disease")

ggplot(data, aes(x=age, y=cholesterol, color=factor(target))) + 
  geom_point(alpha=0.5) +
  facet_wrap(~sex) +
  labs(title="Age vs. Cholesterol with Facets for Sex", x="Age", y="Cholesterol", color="Heart Disease")

data$age_group <- cut(data$age, breaks=c(20, 30, 40, 50, 60, 70, 80), right=FALSE)
ggplot(data, aes(x=age_group, y=cholesterol, fill=factor(target))) + 
  geom_boxplot() +
  facet_wrap(~sex) +
  labs(title="Cholesterol by Age Group and Sex Colored by Heart Disease Presence", x="Age Group", y="Cholesterol", fill="Heart Disease")

subset_data <- data[, c("age", "cholesterol", "sex", "target")]
cor_matrix <- cor(subset_data)
melted_cor_matrix <- melt(cor_matrix)
ggplot(data = melted_cor_matrix, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  geom_text(aes(label=round(value, 2)), color="white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1)) +
  theme_minimal() + 
  labs(title="Heatmap of Correlations between Age, Cholesterol, Sex, and Target", x="Variables", y="Variables", fill="Correlation")

subset_data <- data[, c("age", "cholesterol", "sex", "target")]
ggpairs(subset_data, aes(color=factor(target), alpha=0.5)) +
  labs(title="Scatter Plot Matrix of Age, Cholesterol, Sex, and Target")

#install.packages("ggExtra")
library(ggExtra)
library(ggplot2)
library(tidyr)
library(dplyr)
# Ensure 'target' and 'sex' are factors
data <- data %>%
  mutate(
    target = as.factor(target),
    sex = as.factor(sex)
  )
ggplot(data, aes(x=age, y=cholesterol, color=target)) + 
  geom_point(alpha=0.6) +
  labs(title="Age vs. Cholesterol Colored by Heart Disease Presence", x="Age", y="Cholesterol", color="Heart Disease")

p <- ggplot(data, aes(x=age, y=cholesterol, color=target)) + 
  geom_point(alpha=0.6) +
  geom_smooth(method="lm", se=FALSE, linetype="dashed") +
  facet_wrap(~sex) +
  labs(title="Age vs. Cholesterol with Regression Lines and Density Plots", x="Age", y="Cholesterol", color="Heart Disease") +
  theme_minimal()

# Adding marginal density plots
ggMarginal(p, type="density", groupColour=TRUE, groupFill=TRUE)

library(ggplot2)
library(reshape2)
library(RColorBrewer)

subset_data <- data[, c("age", "cholesterol", "sex", "target")]
cor_matrix <- cor(subset_data)
melted_cor_matrix <- melt(cor_matrix)

ggplot(data = melted_cor_matrix, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color="white") +
  geom_text(aes(label=round(value, 2)), color="black", size=4) +
  scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0, limit=c(-1,1), space="Lab", name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle=45, vjust=1, size=12, hjust=1)) +
  coord_fixed() +
  labs(title="Heatmap of Correlations with Detailed Annotations", x="Variables", y="Variables", fill="Correlation")


# Ensure 'target' and 'sex' are factors
data$target <- as.factor(data$target)
data$sex <- as.factor(data$sex)

ggparcoord(data, columns=c("age", "cholesterol", "sex", "target"), groupColumn="target", scale="std", alphaLines=0.5, showPoints=TRUE) +
  scale_color_manual(values=c("0"="blue", "1"="red")) +
  theme_minimal() +
  labs(title="Parallel Coordinates Plot with Color Encoding", x="Variables", y="Standardized Values", color="Heart Disease")


# Prepare data for radar chart
data_radar <- data %>%
  dplyr::select(age, cholesterol, target) %>%
  group_by(target) %>%
  summarize(across(everything(), mean, na.rm = TRUE))

# Convert 'target' back to numeric for the radar chart
data_radar <- data_radar %>%
  mutate(target = as.numeric(as.character(target)))

# Ensure all columns are numeric
data_radar <- data_radar %>%
  mutate(across(everything(), as.numeric))

# Verify the structure of data_radar
str(data_radar)

# Add the required upper and lower bounds for radar chart
max_values <- apply(data_radar, 2, max, na.rm = TRUE)
min_values <- apply(data_radar, 2, min, na.rm = TRUE)

# Add bounds to the data
data_radar <- rbind(max_values, min_values, data_radar)

# Ensure all columns are numeric
data_radar <- data_radar %>%
  mutate(across(everything(), as.numeric))

# Verify the structure of data_radar again
str(data_radar)

# Create the radar chart
radarchart(data_radar, axistype=1, pcol=c("#1f77b4","#ff7f0e"), pfcol=c("#1f77b480","#ff7f0e80"), plwd=2, 
           cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0, 100, by=20), cglwd=0.8,
           vlcex=0.8, title="Radar Chart Comparing Groups")
legend(x=1.1, y=1, legend = c("No Heart Disease","Heart Disease"), bty = "n", pch=20, col=c("#1f77b4","#ff7f0e"), text.col = "black", cex=1.2, pt.cex=3)

# Ensure numeric columns for the parallel coordinates plot
data_parallel <- data %>%
  dplyr::select(age, cholesterol, target) %>%
  mutate(across(everything(), as.numeric))

# Create the parallel coordinates plot
ggparcoord(data_parallel, columns=1:5, groupColumn=6, scale="std", alphaLines=0.5, showPoints=TRUE) +
  scale_color_manual(values=c("0"="blue", "1"="red")) +
  theme_minimal() +
  labs(title="Parallel Coordinates Plot with Color Encoding", x="Variables", y="Standardized Values", color="Heart Disease")


# Box Plot for Age vs Target
ggplot(data, aes(x = as.factor(target), y = age)) +
  geom_boxplot(fill = 'yellow', color = 'black', alpha = 0.7) +
  labs(title = 'Box Plot of Age by Target', x = 'Target (0 = No Heart Disease, 1 = Heart Disease)', y = 'Age')

# Scatter Plot of Age vs Cholesterol
ggplot(data, aes(x = age, y = cholesterol, color = as.factor(target))) +
  geom_point(alpha = 0.7) +
  labs(title = 'Scatter Plot of Age vs Cholesterol', x = 'Age', y = 'Cholesterol') +
  scale_color_discrete(name = 'Target')

# Define age groups
data <- data %>%
  mutate(age_group = case_when(
    age < 30 ~ '<30',
    age >= 30 & age < 40 ~ '30-39',
    age >= 40 & age < 50 ~ '40-49',
    age >= 50 & age < 60 ~ '50-59',
    age >= 60 ~ '60+'
  ))

# Count of Age Groups
ggplot(data, aes(x = age_group)) +
  geom_bar(fill = 'blue', color = 'black', alpha = 0.7) +
  labs(title = 'Count of Age Groups', x = 'Age Group', y = 'Count')

# Gender Distribution within Age Groups
ggplot(data, aes(x = age_group, fill = as.factor(sex))) +
  geom_bar(position = 'dodge') +
  labs(title = 'Gender Distribution within Age Groups', x = 'Age Group', y = 'Count', fill = 'Sex (0 = Female, 1 = Male)')

# Chest Pain Type Distribution within Age Groups
ggplot(data, aes(x = age_group, fill = as.factor(chest.pain.type))) +
  geom_bar(position = 'dodge') +
  labs(title = 'Chest Pain Type Distribution within Age Groups', x = 'Age Group', y = 'Count', fill = 'Chest Pain Type')

# Resting Blood Pressure by Age Group
ggplot(data, aes(x = age_group, y = resting.bp.s)) +
  geom_boxplot(fill = 'purple', color = 'black', alpha = 0.7) +
  labs(title = 'Resting Blood Pressure by Age Group', x = 'Age Group', y = 'Resting Blood Pressure (systolic)')

# Cholesterol Levels by Age Group
ggplot(data, aes(x = age_group, y = cholesterol)) +
  geom_boxplot(fill = 'red', color = 'black', alpha = 0.7) +
  labs(title = 'Cholesterol Levels by Age Group', x = 'Age Group', y = 'Cholesterol')

# Max Heart Rate by Age Group
ggplot(data, aes(x = age_group, y = max.heart.rate)) +
  geom_boxplot(fill = 'green', color = 'black', alpha = 0.7) +
  labs(title = 'Max Heart Rate by Age Group', x = 'Age Group', y = 'Max Heart Rate')

# Exercise Angina within Age Groups
ggplot(data, aes(x = age_group, fill = as.factor(exercise.angina))) +
  geom_bar(position = 'dodge') +
  labs(title = 'Exercise Angina within Age Groups', x = 'Age Group', y = 'Count', fill = 'Exercise Angina (0 = No, 1 = Yes)')

# Target Distribution within Age Groups
ggplot(data, aes(x = age_group, fill = as.factor(target))) +
  geom_bar(position = 'dodge') +
  labs(title = 'Target Distribution within Age Groups', x = 'Age Group', y = 'Count', fill = 'Target (0 = No Heart Disease, 1 = Heart Disease)')

# Scatter Plot of Age vs Cholesterol by Age Group
ggplot(data, aes(x = age, y = cholesterol, color = age_group)) +
  geom_point(alpha = 0.7) +
  labs(title = 'Scatter Plot of Age vs Cholesterol by Age Group', x = 'Age', y = 'Cholesterol') +
  scale_color_discrete(name = 'Age Group')

# Calculate average values within each age group
avg_data <- data %>%
  group_by(age_group) %>%
  summarise(
    avg_resting_bp = mean(resting.bp.s, na.rm = TRUE),
    avg_cholesterol = mean(cholesterol, na.rm = TRUE),
    avg_max_heart_rate = mean(max.heart.rate, na.rm = TRUE),
    avg_oldpeak = mean(oldpeak, na.rm = TRUE)
  )

# Average Resting Blood Pressure by Age Group
ggplot(avg_data, aes(x = age_group, y = avg_resting_bp)) +
  geom_bar(stat = "identity", fill = 'blue', color = 'black', alpha = 0.7) +
  labs(title = 'Average Resting Blood Pressure by Age Group', x = 'Age Group', y = 'Average Resting Blood Pressure (systolic)')

# Average Cholesterol Levels by Age Group
ggplot(avg_data, aes(x = age_group, y = avg_cholesterol)) +
  geom_bar(stat = "identity", fill = 'red', color = 'black', alpha = 0.7) +
  labs(title = 'Average Cholesterol Levels by Age Group', x = 'Age Group', y = 'Average Cholesterol')

# Average Maximum Heart Rate by Age Group
ggplot(avg_data, aes(x = age_group, y = avg_max_heart_rate)) +
  geom_bar(stat = "identity", fill = 'green', color = 'black', alpha = 0.7) +
  labs(title = 'Average Maximum Heart Rate by Age Group', x = 'Age Group', y = 'Average Max Heart Rate')

# Calculate average age by chest pain type
avg_age_chest_pain <- data %>%
  group_by(chest.pain.type) %>%
  summarise(avg_age = mean(age, na.rm = TRUE))

# Average Age by Chest Pain Type
ggplot(avg_age_chest_pain, aes(x = as.factor(chest.pain.type), y = avg_age, fill = as.factor(chest.pain.type))) +
  geom_bar(stat = "identity", color = 'black', alpha = 0.7) +
  labs(title = 'Average Age by Chest Pain Type', x = 'Chest Pain Type', y = 'Average Age') +
  scale_fill_brewer(palette = "Pastel1")

# Scatter Plot of Age vs Target with Trend Line
ggplot(data, aes(x = age, y = target, color = age_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = 'Scatter Plot of Age vs Target with Trend Line', x = 'Age', y = 'Target (0 = No Heart Disease, 1 = Heart Disease)') +
  scale_color_brewer(palette = "Pastel1")

# Scatter Plot of Age vs Fasting Blood Sugar with Trend Line
ggplot(data, aes(x = age, y = fasting.blood.sugar, color = age_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = 'Scatter Plot of Age vs Fasting Blood Sugar with Trend Line', x = 'Age', y = 'Fasting Blood Sugar') +
  scale_color_brewer(palette = "Set1")

# Scatter Plot of Age vs Resting Blood Pressure with Trend Line
ggplot(data, aes(x = age, y = resting.bp.s, color = age_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = 'Scatter Plot of Age vs Resting Blood Pressure with Trend Line', x = 'Age', y = 'Resting Blood Pressure (systolic)') +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# Scatter Plot of Age vs Resting ECG with Trend Line
ggplot(data, aes(x = age, y = resting.ecg, color = age_group)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = 'Scatter Plot of Age vs Resting ECG with Trend Line', x = 'Age', y = 'Resting ECG') +
  scale_color_brewer(palette = "Set2") +
  theme_minimal() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# Age group distribution
age_group_dist <- data %>%
  group_by(age_group) %>%
  summarise(count = n()) %>%
  mutate(percent = count / sum(count) * 100)

# Donut chart for age group distribution
ggplot(age_group_dist, aes(x = 2, y = percent, fill = age_group)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  theme_void() +
  theme(legend.position = "right") +
  labs(title = 'Age Group Distribution') +
  scale_fill_brewer(palette = "Set3") +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            position = position_stack(vjust = 0.5), 
            color = "white")

# Gender distribution within age groups
gender_age_dist <- data %>%
  group_by(age_group, sex) %>%
  summarise(count = n()) %>%
  mutate(percent = count / sum(count) * 100)

# Donut chart for gender distribution within age groups
ggplot(gender_age_dist, aes(x = 2, y = percent, fill = as.factor(sex))) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  facet_wrap(~ age_group) +
  theme_void() +
  theme(legend.position = "right") +
  labs(title = 'Gender Distribution within Age Groups', fill = 'Sex (0 = Female, 1 = Male)') +
  scale_fill_brewer(palette = "Set1") +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            position = position_stack(vjust = 0.5), 
            color = "white")

# Chest pain type distribution within age groups
chest_pain_age_dist <- data %>%
  group_by(age_group, chest.pain.type) %>%
  summarise(count = n()) %>%
  mutate(percent = count / sum(count) * 100)

# Donut chart for chest pain type distribution within age groups
ggplot(chest_pain_age_dist, aes(x = 2, y = percent, fill = as.factor(chest.pain.type))) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  facet_wrap(~ age_group) +
  theme_void() +
  theme(legend.position = "right") +
  labs(title = 'Chest Pain Type Distribution within Age Groups', fill = 'Chest Pain Type') +
  scale_fill_brewer(palette = "Set2") +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            position = position_stack(vjust = 0.5), 
            color = "white")


# Exclude individuals under 30 years old
data <- data %>%
  filter(age >= 30)

# Define age groups
data <- data %>%
  mutate(age_group = case_when(
    age >= 30 & age < 40 ~ '30-39',
    age >= 40 & age < 50 ~ '40-49',
    age >= 50 & age < 60 ~ '50-59',
    age >= 60 ~ '60+'
  ))

# Target distribution within age groups
target_age_dist <- data %>%
  group_by(age_group, target) %>%
  summarise(count = n()) %>%
  mutate(percent = count / sum(count) * 100)

# Donut chart for target distribution within age groups
ggplot(target_age_dist, aes(x = 2, y = percent, fill = as.factor(target))) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  facet_wrap(~ age_group) +
  theme_void() +
  theme(legend.position = "right") +
  labs(title = 'Heart Disease Distribution within Age Groups', fill = 'Target (0 = No Heart Disease, 1 = Heart Disease)') +
  scale_fill_brewer(palette = "Set3") +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            position = position_stack(vjust = 0.5), 
            color = "white")

