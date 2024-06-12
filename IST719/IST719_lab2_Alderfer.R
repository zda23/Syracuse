
#lab 2
#Zane Alderfer

fname <- file.choose()
sales <- read.csv(file=fname
                  , header=TRUE
                  , stringsAsFactors = F)

View(sales)
str(sales)
colnames(sales)

sales$expenses[1:10]
sales$income[1:10]

# relationships of continuous by continuous data
plot(sales$expenses, sales$income, main = "scatter")
plot(sales$expenses,sales$income,col="orange",
     xlab="Expenses",ylab="income",
     main="Relationship between expenses and income")

# how do we see what kind of relationship it is? TRENDLINE
abline(lm(sales$income ~ sales$expenses), col="red", lwd = 3)
# abline niftyness
abline(h = 400, col = "blue")
abline(v = 9, col = "blue")
rug(x = sales$income, side = 2, col = "orange")
rug(x = sales$expenses, side = 1, col = "orange")

sales$type[1:10]
boxplot(sales$expenses ~ sales$type)
# ALSO THIS IS A MULTI DIMENSION PLOT
tapply(sales$expenses, list(sales$type), mean)

unique(sales$wine) # category
head(sales$units.sold)
head(sales$type) 

boxplot(sales$units.sold~sales$type,xlab="Wine Type",ylab="Units Sold",
        col=c("palegreen","paleturquoise1"),main="2 dim box plot")

abline(h=mean(sales$units.sold), lty=2,col="red")
units.by.type<- aggregate(sales$units.sold,list(type=sales$type), FUN = sum)
units.by.type

boxplot(sales$units.sold~sales$wine,xlab="Wine",ylab="Units
Sold",col=c("palegreen","paleturquoise1"),
        main="2 dim box plot")
units.by.wine<- aggregate(sales$units.sold,list(wine=sales$wine), FUN = sum)
units.by.wine

sales$units.sold[1:10]
sales$rep.region[1:10]

options(scipen=999)
par(mar=c(5,8,4,2)) 
trec<-tapply(sales$income,sales$year,sum) # total income by year
trec
plot(x,trec,type="l",ylim=c(0,max(trec)),las=2,xlab="Year",
     ylab="",col="deepskyblue2",lwd=3,main="Total Income over time by Year")
mtext(text="income",side=2,line=4)

plot(x,trec,type="l",ylim=c(min(trec),max(trec)),las=2,xlab="Year",
     ylab="",col="deepskyblue2",lwd=3,main="Total Income over time by Year")
mtext(text="Income",side=2,line=4)

M <- tapply(sales$income,list(sales$rep.region,sales$year),sum)

x<-as.numeric(colnames(M))
x

plot(M[1,], type = "l")
x <- as.numeric(colnames(M))
options(scipen=999)
plot(x, M[1,], type = "l", col="red", lwd=2
     , ylab = "recipts in dollars"
     , xlab = "years"
     , ylim = c(0,max(M)), bty = "n")
lines(x, M[2,], col="blue", lwd=2)
lines(x, M[3,], col="orange", lwd=2, lty=2)
lines(x, M[4,], col="brown", lwd=2, lty=2)
lines(x, M[5,], col="green", lwd=2, lty=2)

legend('bottomleft', legend = rownames(M), lwd=2
      , lty=1, col=c('red', 'blue', 'orange', 'brown',' green'), bty='n', cex=.75)

fname.art <- file.choose()
art.1 <- read.csv(file = fname.art
                  , header = TRUE
                  , stringsAsFactors = TRUE)

table(art.1$paper, art.1$paper.type)
colnames(art.1)
water <- art.1[art.1$paper == "watercolor", ]
barplot(tapply(water$units.sold, list(water$paper.type), sum))
art.2 <- read.csv(file = fname.art
                  , header = TRUE
                  , stringsAsFactors = FALSE)
water.2 <- art.2[art.1$paper == "watercolor", ]
barplot(tapply(water.2$units.sold, list(water.2$paper.type), sum))
art.1$paper.type
class(art.1$paper.type)
# factors store numbers
art.2$paper.type
