my.dir = "/Users/zanealderfer/Downloads/"
tweets = read.csv(file=paste0(my.dir, "climatetweets_useforlecture_25k.csv")
                  , header = TRUE
                  , quote = "\""
                  , stringsAsFactors = FALSE)
my.media = tweets$media
table(my.media)
my.media[my.media==""] = "text only"
my.media = gsub("\\|photo", "", my.media)
100 * round(table(my.media)/sum(table(my.media)), 4)
pie(100 * round(table(my.media)/sum(table(my.media)), 4))
tweets$created_at[1:3]

conversion.string = "%a %b %d %H:%M:%S +0000 %Y"

tmp = strptime(tweets$created_at, conversion.string)
class(tmp)
any(is.na(tmp))


rm(tmp)
tweets$date = strptime(tweets$created_at, conversion.string)

tmp = "10AM and 27 minutes, on June 22, 1999"
str
strptime(tmp, "%H%p and %M minutes, on %B %d, %Y")




min(tweets$date)
,sx(tweets$date)
max(tweets$date)
range(tweets$date)
summary(tweets$date)

difftime(min(tweets$date), max(tweets$date))
difftime(min(tweets$date), max(tweets$date), units = "min")
difftime(min(tweets$date), max(tweets$date), units = "weeks")

library(lubridate)

wday(tweets$date[1:3], label = TRUE, abbr = TRUE)

barplot(table(wday(tweets$date, label = TRUE, abbr = TRUE)))

tmp = tweets$user_utc_offset

tweets$date[7:10] + tmp[7:10]

known.times  = tweets$date + tweets$user_utc_offset



index = which(is.na(known.times))

known.times = known.times[-index]

barplot(table(hour(known.times)))

start.date = as.POSIXct(("2016-06-24 23:59:59"))
end.date = as.POSIXct(("2016-06-26 00:00:00"))

index = which((tweets$date > start.date) & (tweets$date < end.date))

tweets.25th = tweets$date[index]
format.Date(tweets.25th, "%Y%m%d%H%M")
tmp.date = as.POSIXct(strptime(format.Date(tweets.25th, "%Y%m%d%H%M"), "%Y%m%d%H%M"))

plot(table(tmp.date))

length(table(tmp.date))

tmp.tab = table(tmp.date)

plot(as.POSIXct(names(tmp.tab)), as.numeric(tmp.tab), type = "h")
class(names(tmp.tab))

x = seq.POSIXt(from = start.date + 1, to = end.date - 1, by = "min")
length(x)
y = rep(0, length(x))
y[match(names(tmp.tab), as.character(x))] = as.numeric(tmp.tab)

plot(x,y, type = "p", pch = 16, cex = .4)

tweets$text[5:10]

library(stringr)

tags = str_extract_all(tweets$text, "#\\S+", simplify = FALSE)     
tags = tags[lengths(tags) > 0]
tags = unlist(tags)
tags

tasg = tolower(tags)
tags = tolower(tags)
tags = gsub("#|[[:punct:]]", "", tags)

tag.tab = sort(table(tags), decreasing = TRUE)
tag.tab[1:10]

zap = which(tag.tab < 3)
tag.tab = tag.tab[-zap]

boxplot(as.numeric(tag.tab))
plot(as.numeric(tag.tab))

df = data.frame(words = names(tag.tab), count = as.numeric(tag.tab), 
                stringsAsFactors = FALSE)

par(mfrow = c(3,3))
plot(df$count, main = "raw")
y = df$count/max(df$count)
plot(y, main = "0 - 1")
plot(df$count^2, main = "^2")
plot(df$count^(1/2), main = "^(1/2)")
plot(df$count^(1/5), main = "^(1/5)")
plot(log10(df$count), main = "log10")
plot(log(df$count), main = "log10")


library(wordcloud)
myPal = colorRampPalette(c("green", "blue", "purple"))

gc()

df

index = which(df$count > 10)
par(mar=c(0,0,0,0), bg = "black")

my.counts = (df$count[index])^(1/2)
wordcloud(df$words[index], my.counts, scale = c(4, .4)
          , min.freq = 1
          , max.words = Inf, random.order = FALSE
          , random.color = FALSE, ordered.colors = TRUE
          , rot.per = 0, colors = myPal(length(df$words[index])))

library(alluvial)
install.packages("alluvial")
library(alluvial)
dat = as.data.frame(Titanic, stringsAsFactors = FALSE)
alluvial(dat[,1:4], freq = dat$Freq)

my.dir = "/Users/zanealderfer/Downloads/"
sales= read.csv(file=paste0(my.dir, "sales (2).csv")
                  , header = TRUE
                  , quote = "\""
                  , stringsAsFactors = FALSE)
sales
alluv.df = aggregate(sales$units.sold, list(sales$rep.region, sales$type)
                     , sum)

colnames(alluv.df) = c("reg", "type", "units.sold")

alluvial(alluv.df[ , 1:2], freq = alluv.df$units.sold)
my.cols = rep("gold", nrow(alluv.df))
my.cols[alluv.df$type == "red"] ='red'
alluvial(alluv.df[ , 1:2], freq = alluv.df$units.sold, col = my.cols)

alluvial(alluv.df[ , 1:2], freq = alluv.df$units.sold
         , col = ifelse(alluv.df$type == "red", "red", "gold"))

options(stringsAsFactors = FALSE)
alluv.df = aggregate(sales$units.sold
                     , list(sales$rep.region
                            , sales$type
                            , sales$wine)
                     , sum)

colnames(alluv.df) = c("reg", "type", "wine", "units.sold")

alluvial(alluv.df[ , 1:3], freq = alluv.df$units.sold
         , col = ifelse(alluv.df$type == "red", "red", "gold"))

library(RColorBrewer)

install.packages("treemap")

library(treemap)
treemap(sales, index = c("rep.region")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "dens"
        , fontsize.labels = 18
        , palette = "Greens")

treemap(sales, index = c("rep.region")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "value"
        , fontsize.labels = 18
        , palette = "OrRd")

treemap(sales, index = c("rep.region", "sales.rep", "type")
        , vSize = "income"
        , vColor = "units.sold"
        , type = "value"
        , fontsize.labels = 18
        , palette = brewer.pal(8, "Set1"))


dat = tapply(sales$units.sold, list(sales$type, sales$rep.region), sum)

barplot(dat, beside = TRUE, col = c("brown", "gold")
        , main = "Units Sold by region by type")
