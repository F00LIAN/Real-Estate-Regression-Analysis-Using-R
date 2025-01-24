# regression analysis in R
setwd("D:/Foundation of Business Analytics/August 2021/Class 3/")

# read in a sample excel sheet
library("readxl")
homes2 = as.data.frame(read_excel("D:\\Foundation of Business Analytics\\August 2021\\Class 3\\homes2.xlsx",sheet="Sheet1"))
attach(homes2)
homes2

# low resolution scatterplot
plot(Floor,Price,type='p')

# slighly higher resolution plot
plot(Floor, Price,frame=F, pch=19, cex=2, font.axis=2, font.lab=2)

# fit a simple linear regression in R
reg1 <- lm(Price ~ Floor)
reg1 # this prints the regression coefficients

# show Y intercept
plot(Floor, Price,frame=F, pch=19, cex=2, font.axis=2, 
     ylim=c(180, 285), xlim=c(0, 3),font.lab=2)


# overlay the regression line on the plot
abline(reg1, col="red")

# illustrate slope effect
plot(Floor, Price, frame=F, pch=19, cex=2, font.axis=2, 
     ylim=c(180, 285), xlim=c(0, 3),font.lab=2)
abline(reg1, col="red")

# fit statistics and model results
summary(reg1)

# extract fitted values and residuals
fits <- fitted(reg1)
resid <- resid(reg1)

# scatterplot with regression line on it
plot(Floor, Price,frame=F, 
     ylim=c(255,285), pch=19, cex=1, font.axis=2, font.lab=2)
abline(reg1, col="red")

# add fitted values
points(Floor, fits, pch=4, col="darkblue", lwd=2)
detach(homes2)
rm(list=ls(all=T))

# read in the real estate data
homes3 <- read.table("real estate data.txt", header=T, sep="\t")
attach(homes3)

# suppress scientific notation
options(scipen=999)

# simple linear regression using only square footage (heated) as the independent variable
reg2 <- lm(salespr ~ sqftheat)
summary(reg2)

# let's plot the actual real estate data
plot(sqftheat, salespr, frame=F, pch=19, font.axis=2, font.lab=2, cex=.5)

# now let's insert the regression line
abline(reg2, col="red", lwd=2)

# what is the expected selling price of a house that has 2000 heated square feet?
coefs <- coef(reg2)
coefs
coefs[1] + coefs[2]*2000


# multiple regression using square feet (heated), square feet (unheated), number of bedrooms, bathrooms, total acreage and home age
reg3 <- lm(salespr ~ bedrooms + acres + age + totbaths + sqftheat + squnheat)
reg3
summary(reg3)

# predict the selling prices of 20 new homes
new.homes <- read.table("20 new homes.txt", header=T, sep="\t")
new.homes

sales.prices <- predict(object=reg3, newdata=new.homes)

# the characteristics of the 20 new homes with the selling prices appended
sales20 <- cbind(new.homes, sales.prices)
sales20

### dummy variables

# let's look at including closesum dummy variable in our model
reg4 <- lm(salespr ~ sqftheat + closesum)
reg4


plot(sqftheat, salespr, frame=F, pch=19, cex=.5, font.axis=2, font.lab=2,
     xlab="Square Feet (Heated)", ylab="Sales Price", xlim=c(1000,3000),
     ylim=c(100000,500000))
coefs3 <- coef(reg4)

# calculate the predicted selling price of the homes in summer vs. not in summer.
sales.no.summer <- coefs3[1] + coefs3[2]*sqftheat
sales.summer    <- coefs3[1] + coefs3[2]*sqftheat + coefs3[3]

#plot the two results
lines(sqftheat, sales.no.summer, lwd=2, col="blue") # homes not sold in summer
lines(sqftheat, sales.summer, lwd=2, col="darkgreen") # homes sold in the summer

# changing the reference group for the dummy variable
close_all_other <- 1 - closesum

# refit your regression
reg5 <- lm(salespr ~ sqftheat + close_all_other)
reg5

# this doesn't work...too many dummy variables!
reg6 <- lm(salespr ~ sqftheat + closesum + close_all_other)
reg6

# multiple dummy variables for spring, summer, and fall
reg7 <- lm(salespr ~ sqftheat + closespr + closesum + closefal)
reg7

# extracting the root mean squared error (regression standard error)
sigma(reg7)

# calculating the root mean squared error by hand
errors.2 <- resid(reg7)^2 # squared errors
N <- length(salespr) # how many observations
K <- length(coef(reg7)) # how many variables in the model including intercept
sqrt(sum(errors.2)/(N - K)) # root MSE or standard error of regression

# look at R-squared value
# the R-squared value is reported in R as: "Multiple R-squared"
summary(reg7)

