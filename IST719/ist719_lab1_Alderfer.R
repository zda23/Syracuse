#IST719 Lab 1

pie(c(7,8,10,12))
x <- c(7,8,10,12)
pie(x, main = "Zane's graph", col = c("blue", "red", "orange", "yellow"), labels = c('a','b','c','d'))
plot(c(1,3,6,4), pch = 16, col = c('red','orange','tan','yellow'), cex =3)
y <- rnorm(n=10)
plot(y, type= 'h', lwd = 5, lend = 2, col = 'orange', main = "change in net worth", 
     xlab = 'time in years', ylab = 'in millions')

n <- 27
my.letters <- sample(letters[1:3], size = n, replace = T)
letters[7:9]
letters[c(8,3,1)]
my.letters
tab <- table(my.letters)
barplot(tab, col = c("brown", "tan","orange"), names.arg = c("sales", "ops", "it"), 
        border = "green", xlab = "departments", ylab = "employees", 
        main = "company employees", horiz = TRUE, las =1, density = 20, angle = c(45,90,12))

x <- rnorm(n=1000, mean =10, sd =1)
hist(x, main = "what is the distribution of x")

boxplot(x, horizontal = T)

x <- rlnorm(n=1000, meanlog = 1, sdlog = 1)
boxplot(x, horizontal = T)
hist(log10(x))
