#install.packages("shiny")
library(shiny)

server = function(input, output) {
  output$myPie = renderPlot({
    pie(c(8,12,3), main = "Hello World")
  })
}

ui = fluidPage(
  mainPanel(plotOutput("myPie"))
)

shinyApp(ui, server)

#install.packages("lubridate")
library(lubridate)

server = function(input, output) {
  
}

ui = fluidPage(
  mainPanel(paste("Zane's shiny app at", now()))
)

shinyApp(ui, server)

fname = "/Users/zanealderfer/Downloads/art-1.csv"

artserver = function(input, output) {
  art = read.csv(fname, header = TRUE, stringsAsFactors = FALSE)
  watercolor.col = "cadetblue1"
  drawing.col = "antiquewhite"
  
  #plotOutput("yearlyReceipts")
  output$yearlyReceipts = renderPlot({
    print("yearlyReceipts:: start")
    my.title = "Number of Sales per Year"
    barplot(table(art$year), main = my.title, border = "white"
            , col = "chartreuse4")
  })
  
  output$storePaper  = renderPlot({
    print("storePaper:: start")
    if (input$store != "None") {
      print(paste("storePaper:: store:", input$store))
      sub.index = which(art$store == input$store)
      tmp.data = art[sub.index, ]
      if (input$year != "All") {
        print(paste("storePaper:: year:", input$year))
        sub.index.2 = which(tmp.data$year == as.numeric(input$year))
        tmp.data = tmp.data[sub.index.2, ]
      }
      
      par(mfrow = c(1,2))
      sales.by.paper = tapply(tmp.data$total.sale
                              , list(tmp.data$paper), sum)
      barplot(sales.by.paper, beside = TRUE, main = "Income by paper type"
              , col = c(watercolor.col, drawing.col)
               , border = NA)
      
      sales.by.rep = tapply(tmp.data$total.sale
                              , list(tmp.data$rep), sum)
      pie(sales.by.rep, border = NA
          , col = terrain.colors(length(sales.by.rep)))

      #pie(table(tmp.data$paper), col = c(watercolor.col, drawing.col)
         # , border = NA)
    }
  })
  output$storeEmployee  = renderPlot({
    print("storeEmployee:: start")
    if (input$store != "None") {
      print(paste("storeEmployee:: store:", input$store))
      sub.index = which(art$store == input$store)
      
      tmp.data = art[sub.index, ]
      if (input$year != "All") {
        print(paste("storeEmployee:: year:", input$year))
        sub.index.2 = which(tmp.data$year == as.numeric(input$year))
        tmp.data = tmp.data[sub.index.2, ]
      }
      
      par(mfrow = c(1,2))
      sales.by.paper = tapply(tmp.data$total.sale
                              , list(tmp.data$paper), sum)
      barplot(sales.by.paper, beside = TRUE, main = "Income by paper type"
              , col = c(watercolor.col, drawing.col)
              , border = NA)
      
      sales.by.rep = tapply(tmp.data$total.sale
                            , list(tmp.data$rep), sum)
      pie(sales.by.rep, border = NA
          , col = terrain.colors(length(sales.by.rep)))
      
      #pie(table(tmp.data$paper), col = c(watercolor.col, drawing.col)
      # , border = NA)
    }
  })
}

artui = fluidPage(
  titlePanel("ACME Art Company Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      plotOutput("yearlyReceipts"),
      selectInput("store", "Select Store:"
                  , choices = c("None", "Portland", "Davenport", "Syracuse", "Dublin")),
      selectInput("year", "Select Year:"
                  , choices = c("All", "2012", "2013", "2014", "2015"))
    ),
     mainPanel(
       plotOutput("storePaper"),
       plotOutput("storeEmployee")
   
    )
  )
)


shinyApp(ui = artui, server = artserver)

#install.packages("leaflet")
library(leaflet)
library(ggmap)
my.dir = "/Users/zanealderfer/Downloads/"
libs = read.csv(paste0(my.dir, "NewYorkLibraries.csv")
                , stringsAsFactors = FALSE
                , quote = "\""
                , header = TRUE)
ny.libs = nrow(libs)

server = function(input, output, session) {
  print("server:: start")
  
  points = eventReactive(input$num.libs, {
    index = sample(1:nrow(libs), input$num.libs)
    addys = paste(libs$ADDRESS[index]
                 , libs$CITY[index], libs$STABR[index], sep = ", ")
    g.codes = geocode(addys, source = "dsk")
    df = data.frame(lon = g.codes$lon, lat = g.codes$lat, addy = addys)
    df 
  }, ignoreNULL = FALSE)
  
  output$mymap = renderLeaflet({
    M = leaflet()
    M = addProviderTiles(M, providers$OpenStreetMap
                         , options = providerTileOptions(noWrap = TRUE))
    df = points()
    addMarkers(M, lng = df[, 1], lat = df[,2], popup = df[,3])
  })
}

ui = fluidPage(
  leafletOutput("mymap"),
  numericInput("num.libs", "Number of Libraries", 10, min = 1, max = ny.libs)
)

shinyApp(ui, server)

server = function(input, output) {
  
  output$plot1 = renderPlot({
    A = sample(LETTERS[3:(2+input$slider2)], input$slider1, replace = TRUE)
    B = sample(letters[12:(input$slider3)], input$slider1, replace = TRUE)
  barplot(table(A,B), beside = TRUE)
  })
  
 output$plot2 = renderPlot({
   C.num.letters = input$slider5
   D.num.letters = input$slider6
   print(C.num.letters, D.num.letters)
   
   C = sample(LETTERS[7:(6+C.num.letters)], input$slider4, replace = TRUE)
   D = sample(letters[17:(16+D.num.letters)], input$slider4, replace = TRUE)
   barplot(table(D,C), beside = TRUE)
 })
}

ui = fluidPage(
  titlePanel("Hello Shiny!"),
  
wellPanel(
  fluidRow(
    column(6, 
    sliderInput("slider1", "P1 Observations", min = 10, max = 1000, value = 20),
    sliderInput("slider2", "A cats", min = 2, max = 5, value = 2),
    sliderInput("slider3", "B cats", min = 2, max = 5, value = 3)
    ),
    
    column(6, 
           sliderInput("slider4", "P1 Observations", min = 10, max = 1000, value = 20),
           sliderInput("slider5", "A cats", min = 2, max = 5, value = 2),
           sliderInput("slider6", "B cats", min = 2, max = 5, value = 3)
    )       
  ),
  fluidRow(
    column(6, plotOutput("plot1")),
    column(6, plotOutput("plot2"))
  )
)
)

shinyApp(ui, server)