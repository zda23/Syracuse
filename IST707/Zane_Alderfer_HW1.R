library(dplyr)
students_data = data_storyteller %>%
  mutate(total_students = rowSums(across(c(`Very Ahead +5`, `Middling +0`, `Behind -1-5`, `More Behind -6-10`, `Very Behind -11`, Completed))))
students_data
summary = students_data %>%
  group_by(School) %>%
  summarize(ave_very_ahead = mean(`Very Ahead +5`), ave_middling = mean(`Middling +0`), ave_behind = mean(`Behind -1-5`), ave_more_behind = mean(`More Behind -6-10`), ave_very_behind = mean(`Very Behind -11`), ave_completed= mean(Completed))
summary
summary_sum = students_data %>%
  group_by(School) %>%
  summarize(sum_very_ahead = sum(`Very Ahead +5`), sum_middling = sum(`Middling +0`), sum_behind = sum(`Behind -1-5`), sum_more_behind = sum(`More Behind -6-10`), sum_very_behind = sum(`Very Behind -11`), sum_completed= sum(Completed))
summary_sum
summary_total = summary_sum %>%
  mutate(students_total = rowSums(across(where(is.numeric)), na.rm=TRUE))
summary_percent = summary_total%>%
  group_by(School) %>%
  summarise(very_ahead_percent = sum_very_ahead/students_total, middling_perc = sum_middling/students_total, behind_perc = sum_behind/students_total, more_behind_perc = sum_more_behind/students_total, very_behind_perc = sum_very_behind/students_total, completed_perc = sum_completed/students_total)
summary_percent
boxplot(students_data[,3:8])
barplot(summary_percent$middling_perc, names.arg = summary_percent$School, xlab = "School", ylab = "Percent of students", col = "blue", main = "Percent of Middling Students by School")
barplot(summary_percent$behind_perc, names.arg = summary_percent$School, xlab = "School", ylab = "Percent of students", col = "red", main = "Percent of Behind Students by School")
barplot(summary_percent$more_behind_perc, names.arg = summary_percent$School, xlab = "School", ylab = "Percent of students", col = "green", main = "Percent of More Behind Students by School")
barplot(summary_percent$very_behind_perc, names.arg = summary_percent$School, xlab = "School", ylab = "Percent of students", col = "purple", main = "Percent of Very Behind Students by School")
barplot(summary_percent$completed_perc, names.arg = summary_percent$School, xlab = "School", ylab = "Percent of students", col = "orange", main = "Percent of Completed Students by School")

