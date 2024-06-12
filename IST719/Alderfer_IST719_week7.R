my.dir = "/Users/zanealderfer/Downloads/"
df = read.csv(file = paste0(my.dir, "MapLectureData.csv")
              , header = TRUE
              , stringsAsFactors = FALSE)

plot(df$x, df$y)
polygon(df$x, df$y, col = "firebrick1", border = NA)

library(maps)
library(mapproj)

map(database = "world")

map("world", regions = 'India')
map("world", regions = 'China')
map("world", regions = c('India', 'Pakistan')
    , fill = TRUE, col = c("orange","brown"))
map("world", regions = 'Finland')    

m = map("state")
m
plot(m$x, m$y)

map("state", fill = TRUE, col = c("orange", "red", "yellow"))
map("county", region = "New York", fill = T, col = terrain.colors(20))

library(rnaturalearth)
install.packages('rnaturalearthhires')
india = ne_states(country = "India")
map(india)

india$name
map(india)

map("world", regions = "India", namefield = "name"
    , region = c("Gujarat", "Rajasthan", "Madhya", "Pradesh")
    , col = c("orangered"))

install.packages('raster')
library(raster)

india = raster::getData("GADM", country = "IND", level =1)
map(india)

india$NAME_2

map(india, namefield = "NAME_1", region = "Gujarat")

fname = paste0(my.dir, "shootings.Rda")
load(fname)

shootings$Total.Number.of.Victims
sort(shootings$State)
tmp.vec = gsub("^\\s+|\\s+$", "", shootings$State)

shootings$State = tmp.vec
agg.dat = aggregate(shootings$Total.Number.of.Victims,
                    list(shootings$State)
                    , sum)
agg.dat

colnames(agg.dat) = c("state","victims")
num.cols = 10
my.color.vec = rev(heat.colors(num.cols))
pie(rep(1, num.cols), col = my.color.vec)

library(plotrix)
agg.dat$index = round(rescale(x = agg.dat$victims, c(1, num.cols)), 0)
agg.dat$color = my.color.vec[agg.dat$index]

m = map("state")
m$names

state.order = match.map(database = "state", regions = agg.dat$state
                        , exact = FALSE, warn = TRUE)

cbind(m$names, agg.dat$state[state.order])

map("state", col = agg.dat$color[state.order], fill = TRUE
    , resolution = 0, lty = 1, projection = "polyconic", border = "tan")


library(ggmap)

libs = read.csv(paste0(my.dir, "NewYorkLibraries.csv")
                , header = TRUE, quote = "\""
                , stringsAsFactors = FALSE)

map("world")
points(0,0, col = "red", cex = 3, pch = 8)
abline(h =43, col = "blue", lty = 3)
abline(v =-76, col = "blue", lty = 3)

us.cities
map("state")
my.cols = rep(rgb(1, .6, .2, .7), length(us.cities$name))
my.cols[us.cities$capital > 0] = rgb(.2, .6, 1, .9)

points(us.cities$long, us.cities$lat, col = my.cols
       , pch = 16
       , cex = rescale(us.cities$pop, c(.5,7)))

geocode("3649 Erie Blvd East, Dewitt, ny", source = "dsk")

table(libs$CITY)
index = which(libs$CITY %in% c("SYRACUSE", "DEWITT", "FAYETTEVILLE"))
addy = paste(libs$ADDRESS[index], libs$CITY[index], libs$STABR[index]
             , sep = ", ")

map("county", "new york", fill = TRUE, col= "orange")
g.codes = geocode(addy, source = "dsk")
points(g.codes$lon, g.codes$lat, col = "blue", cex = 1.1, pch = 16)


library(rworldmap)
countries = read.delim(paste0(my.dir, "countries.csv")
                       , quote = "\""
                       , header = TRUE
                       , sep = ";"
                       , stringsAsFactors = FALSE)
range(countries$Life.expectancy)
#zap = which(countries$Life.expectancy == 0.0)
countries = countries[-zap, ]
rm(zap)

num.cat = 10

iso3.codes = tapply(countries$Country..en.
                   , 1:length(countries$Country..en.)
                   , rwmGetISO3)
iso3.codes
df = data.frame(country = iso3.codes, labels = countries$Country..en.
                , life = countries$Life.expectancy)

df.map = joinCountryData2Map(df, joinCode = "ISO3"
                             , nameJoinColumn = "country")

par(mar = c(0,0,1,0))

mapCountryData(df.map
               , nameColumnToPlot = "life"
               , numCats = num.cat
               , catMethod = c("pretty", "fixedwidth", "diverging", "quantiles")[4]
               , colourPalette = colorRampPalette(
                 c("orangered", "palegoldenrod", "forestgreen")
              )(num.cat)
              , oceanCol = "royalblue4"
              , borderCol = "peachpuff4"
              , mapTitle = "Life Expectancy"
              )

library(ggmap)
reported = read.csv(paste0(my.dir, "IndiaReportedRapes.csv")
                    , header = TRUE, quote = "\""
                    , stringsAsFactors = FALSE)

crimes = aggregate(reported$Cases, list(reported$Area_Name), sum)
colnames(crimes) = c('id', 'ReportedRapes')
crimes[order(crimes$ReportedRapes), ]


my.map = merge(x = map, y = crimes, by = "id")
ggplot() + geom_map(data = my.map, map = my.map) +
  aes(x = long, y = lat, map_id = id, group = group
      , fill = ReportedRapes) +
  theme_minimal() + ggtitle("Reported Rapes in India")


library(stringr)

