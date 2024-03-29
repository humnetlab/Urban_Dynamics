
library(EpiEstim)
library(ggplot2)
library(incidence)

dataPath <- "data/Spain_covid/"

cities_spain <- c("Madrid", "Barcelona", "Valencia", "Alicante", "Coruna",
                "Zaragoza", "Sevilla", "Malaga", "Bilbao", "SantaCruz", "Granada")

# load province data
covid_data = data.frame()
for (city in cities_spain) {
  cityData = read.csv(paste(dataPath, "covid_", city, "_202011.csv", sep=""), header=TRUE, sep=",")
  cityData$Date = cityData$Province
  cityData$Province = city
  covid_data = rbind(covid_data, cityData)
}
  
covid_data$dates= as.Date(covid_data$Date, format= "%Y-%m-%d")

for (cityIdx in seq(1,11)){
  targetCity <- cities_spain[cityIdx]
  covid_city <- covid_data[which(covid_data$Province==targetCity),]
  
  covid_city$I <- c(c(covid_city$Cases[1]), diff(as.matrix(covid_city$Cases)))
  
  covid_city$I[covid_city$I < 0] <- 0
  
  # save data
  covid_city$city <- targetCity
  
  # save res to csv
  write.csv(covid_city[,c("city", "dates" ,"Cases", "I")], file=paste(dataPath, "R0_", targetCity, "_incidence.csv", sep=""), row.names=FALSE, quote=FALSE)
  
  plot(as.incidence(covid_city$I, dates = covid_city$dates))
  
  test <- covid_city[,c("I", "dates")]
  # test$Ismooth = as.vector(smooth(test$I))
  # Estimating R on sliding weekly windows, with a parametric serial interval
  T <- nrow(test)
  t_start <- seq(2, T-6) # starting at 2 as conditional on the past observations
  t_end <- t_start + 6 # adding 6 to get 7-day windows as bounds included in window
  
  res_parametric_si <- estimate_R(test, 
                                  method="parametric_si",
                                  config = make_config(list(
                                    t_start = t_start,
                                    t_end = t_end,
                                    mean_si = 2.6, 
                                    std_si = 2.0))
  )
  
  
  head(res_parametric_si$R)
  
  # p <- ggplot()
  plot(res_parametric_si, legend = FALSE)
  
  res <- cbind(as.character(res_parametric_si$dates[res_parametric_si$R$t_end]), res_parametric_si$R$`Mean(R)`,
               res_parametric_si$R$`Quantile.0.05(R)`, res_parametric_si$R$`Quantile.0.95(R)`,
               res_parametric_si$R$`Quantile.0.025(R)`, res_parametric_si$R$`Quantile.0.975(R)`)
  res <- as.data.frame(res)
  colnames(res) <- c("dates", "R0", "Q05", "Q95", "Q025", "Q975")
  res$city <- targetCity
  # save res to csv
  write.csv(res, file=paste(dataPath, "R0_", targetCity, "_range.csv", sep=""), row.names=FALSE, quote=FALSE)
}

# =================== End ================== #
