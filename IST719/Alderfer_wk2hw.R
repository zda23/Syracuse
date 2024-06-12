hotdogs <- read.csv("/Users/zanealderfer/Downloads/hot-dog-contest-winners.csv", 
                    sep = ",", header=TRUE)

fill_colors <- c()
for (i in 1:length(hotdogs$New.record)) {
  if (hotdogs$New.record[i] == 1) {
    fill_colors <- c(fill_colors, "#821122")
  } else {
    fill_colors <- c(fill_colors, "#cccccc")
  }
}
barplot(hotdogs$Dogs.eaten, 
        main = "Nathan's Hot Dog Eating Contest Results, 1980-2010",
        names.arg=hotdogs$Year, 
        col=fill_colors,
        border=NA, 
        space=0.3, 
        xlab="Year",
        ylab="Hot dogs and buns (HBD) eaten")

hot_dog_places <- read.csv("http://datasets.flowingdata.com/hot-dog-places.csv", 
                           sep=",", 
                           header=TRUE)
hot_dog_matrix <- as.matrix(hot_dog_places)

barplot(hot_dog_matrix,
        border = NA,
        space = .25,
        ylim=c(0,200),
        xlab="Year",
        ylab="Hot dogs and buns (HDBs) eaten",
        main = "Hot Dog Eating Contest Results 1980-2010")

subscribers <- read.csv("http://datasets.flowingdata.com/flowingdata_subscribers.csv",
                        sep=",",
                        header=TRUE)
plot(subscribers$Subscribers,
     type="h",
     ylim=c(0,30000),
     xlab="Day",
     ylab="Subscribers")
points(subscribers$Subscribers, pch=19, col="black")

population <- read.csv("http://datasets.flowingdata.com/world-population.csv",
                       sep=",",
                       header=TRUE)
plot(population$Year, population$Population,
     type = "l",
     ylim=c(0, 7000000000),
     xlab="Year",
     ylab="Population")

postage <- read.csv("http://datasets.flowingdata.com/us-postage.csv", 
                    sep=",",
                    header=TRUE)
plot(postage$Year, postage$Price, 
     type = "s", 
     main ="US Postage Rates for Letters, First Ounce, 1991-2010",
     xlab="Year",
     ylab="Postage Rate (Dollars)")

library(ggplot2)
fname.art <- file.choose()
art.1 <- read.csv(file = fname.art
                  , header = TRUE
                  , stringsAsFactors = TRUE)

par(mfrow=c(2,2))

hist(art.1$total.sale, main="Distribution of Sales", xlab="Total Sales", ylab="Frequency", col="skyblue")

plot(density(art.1$total.sale), main="Distribution of Sales", xlab="Total Sales", ylab="Density", col="orange")

drawing_paper <- subset(art.1, paper == "drawing")

watercolor_paper <- subset(art.1, paper == "watercolor")

boxplot(total.sale ~ paper, data=drawing_paper, main="distribution of the totals
sales for drawing paper", xlab="Paper", ylab="Total Sales", col="lightgreen")

boxplot(total.sale ~ paper, data=watercolor_paper, main="distribution of the
totals sales for watercolor paper", xlab="Paper", ylab="Total Sales", col="red")


par(mfrow=c(1,3))

plot(art.1$unit.price, art.1$units.sold, pch=19, col="darkgreen",
     xlab="Unit Price", ylab="Units Sold", main="Relationship between Unit Price and Units Sold")

units_sold <- aggregate(units.sold ~ paper, data=art.1, FUN=sum)
barplot(units_sold$units.sold, names.arg=units_sold$paper, col="steelblue",
        xlab="Paper", ylab="Total Units Sold", main="Total Units Sold by Paper")

art.1$Income <- art.1$unit.price * art.1$units.sold
total_income <- aggregate(Income ~ paper, data=art.1, FUN=sum)
barplot(total_income$Income, names.arg=total_income$paper, col="salmon",
        xlab="Paper", ylab="Total Income", main="Total Income by Paper")
