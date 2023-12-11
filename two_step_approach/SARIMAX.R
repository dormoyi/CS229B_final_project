top_level_train <- read.csv("data/temp/top_level_train.csv")
top_level_train[, "is_holiday"] <- as.numeric(top_level_train[, "is_holiday"] == "True")
top_level_test <- read.csv("data/temp/top_level_test.csv")
top_level_test[, "is_holiday"] <- as.numeric(top_level_test[, "is_holiday"] == "True")
library(forecast)

p <- 1
d <- 1
q <- 1
P <- 1
D <- 1
Q <- 1
m <- 24

cols_to_include = c("avg_humidity", "avg_temperature", "year", "cos_month", "sin_month", "is_holiday", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6")
endog <- top_level_train$load
exog <- top_level_train[, which(names(top_level_train) %in% cols_to_include)]
exog_test <- top_level_test[, which(names(top_level_test) %in% cols_to_include)]
mean_endog <- mean(endog)
std_endog <- sd(endog)
mean_exog <- colMeans(exog)
std_exog <- apply(exog, 2, sd)
standardized_endog <- (endog - mean_endog)/std_endog
standardized_exog <- scale(exog, center = mean_exog, scale = std_exog)
standardized_exog_test <- scale(exog_test, center = mean_exog, scale = std_exog)

# Create a time series object
ts_endog <- ts(standardized_endog, frequency = m)

# Fit SARIMA model
top_model <- Arima(ts_endog, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = m), xreg = standardized_exog, method="CSS")

# Predict SARIMA model

y_pred <- (forecast(top_model, xreg = standardized_exog_test)$mean)*std_endog + mean_endog
y_test = top_level_test$load
diff <- as.numeric(y_pred) - y_test
rmse <- sqrt(mean(y_pred[1:24]-y_test[1:24])^2)
