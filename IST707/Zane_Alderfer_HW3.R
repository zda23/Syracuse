library(plyr)
library(dplyr)
library(arules)
library(readr)
#install.packages("arulesViz")
library(arulesViz)

bankdata <- read_csv("/Users/zanealderfer/Downloads/bankdata_csv_all.csv")

bankdata <- bankdata %>%
  select(-id) %>%
  mutate_if(is.character, funs(as.factor)) %>%
  mutate_if(is.numeric, funs(discretize))

str(bankdata)
colSums(is.na(bankdata))


summary(bankdata$income)
summary(bankdata$age)
summary(bankdata$sex)
summary(bankdata$married)
summary(bankdata$children)
summary(bankdata$car)
summary(bankdata$region)
summary(bankdata$save_act)
summary(bankdata$current_act)
summary(bankdata$mortgage)
summary(bankdata$pep)

rules = apriori(bankdata, parameter = list(supp = 0.065, conf = 0.92, maxlen = 6))
rules = sort(rules, decreasing = TRUE, by = "lift")
inspect(rules[1:5])
inspect(rules[1:12])
inspect(rules[13:23])
summary(rules)

rules = apriori(bankdata, parameter = list(supp = 0.06, conf = 0.9, maxlen = 6))
rules = sort(rules, decreasing = TRUE, by = "lift")
summary(rules)
plot(rules, method = "grouped")
plot(rules, method = "paracoord", control = list(reorder = TRUE))
