energydata_check = read.csv("/Users/zanealderfer/Downloads/energy_dataset.csv")
weatherdata_check = read.csv("/Users/zanealderfer/Downloads/weather_features.csv")
view(energydata_check)
class(energydata$time)
energydata$time <- gsub(x = energydata$time,  
                       pattern = "January",  
                       replacement = "Jan")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-02-*",  
                        replacement = "Feb")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-03-*",  
                        replacement = "Mar")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-04-*",  
                        replacement = "Apr")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-05-*",  
                        replacement = "May")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-06-*",  
                        replacement = "Jun")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-07-*",  
                        replacement = "Jul")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-08-*",  
                        replacement = "Aug")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-09-*",  
                        replacement = "Sep")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-10-*",  
                        replacement = "Oct")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-11-*",  
                        replacement = "Nov")
energydata$time <- gsub(x = energydata$time,  
                        pattern = "*-12-*",  
                        replacement = "Dec")
View(weatherdata)
class(weatherdata)
library(dplyr)
library(tidyverse)
colnames(weatherdata)
"time" = names(weatherdata)[names(weatherdata) == "dt_iso"]
View(weatherdata)
names(weatherdata)[0] = "time"
View(weatherdata)
weatherdata1 = as_tibble(weatherdata)
weatherdata1
weatherdata1 %>%
  rename(time = dt_iso)
weatherdata1
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-01-*",  
                        replacement = "Jan")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-02-*",  
                        replacement = "Feb")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-03-*",  
                        replacement = "Mar")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-04-*",  
                        replacement = "Apr")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-05-*",  
                        replacement = "May")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-06-*",  
                        replacement = "Jun")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-07-*",  
                        replacement = "Jul")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-08-*",  
                        replacement = "Aug")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-09-*",  
                        replacement = "Sep")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-10-*",  
                        replacement = "Oct")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-11-*",  
                        replacement = "Nov")
weatherdata$dt_iso <- gsub(x = weatherdata$dt_iso,  
                        pattern = "*-12-*",  
                        replacement = "Dec")

view(energydata$time)
energydata$time = substr(energydata$time, 5,7)
weatherdata$dt_iso = substr(weatherdata$dt_iso, 5,7)
weatherdata %>%
  rename(time = dt_iso)
view(weatherdata$dt_iso)
view(energydata$time)
view(energydata)
write.csv(weatherdata, "/Users/zanealderfer/Downloads/weatherdataupdated.csv",row.names = FALSE)
