# regression analysis in R

install.packages("readxl")

# read in a sample excel sheet
library("readxl")
homes2 = as.data.frame(read_excel("D:/data/homes2.xlsx", sheet="Sheet1"))
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
homes3 <- read.table("D:/data/real estate data.txt", header=T, sep="\t")
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
new.homes <- read.table("D:/data/20 new homes.txt", header=T, sep="\t")
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

###################### Advanced Regression Techniques #################
# run a new regression model
homes3 <- read.table("D:/data/real estate data part two.txt", header=T, sep="\t")
attach(homes3)

# suppress scientific notation
options(scipen=999)
par(mfrow=c(1,1))

# rerun a new regression for sales prices
# multiple regression using square feet (heated), square feet (unheated), number of bedrooms, bathrooms, total acreage and home age
reg3 <- lm(salespr ~ bedrooms + acres + age + totbaths + sqftheat + squnheat)
reg3
summary(reg3)

# Identifying Influential Observations with Cook's Distance
# Calculate Cook's distance
cooksd <- cooks.distance(reg3)

# Plot Cook's distance with a reference line at 0.5
plot(cooksd, frame=FALSE)
abline(h=0.5, lty=2)

# Flag values > 0.5
bad.values <- ifelse(cooksd > 0.5, 1, 0)
table(bad.values)  # Count how many observations exceed 0.5

# Combine the variables into one dataset
my.data.set <- cbind(salespr, bedrooms, acres, age, totbaths, sqftheat, squnheat)

# Inspect the flagged row(s)
my.data.set[bad.values == 1,]

# Compare flagged values to natural quantiles
apply(my.data.set, 2, quantile)

# Evaluating Model EFunctionality on Unseen Test Data
# hold out sample
summary(reg3)
detach(homes3)

K <- 500
N <- nrow(my.data.set)

train <- as.data.frame(my.data.set[1:K,])
test <- as.data.frame(my.data.set[(K+1):N,])
test.Y <- test[,1]
test.X <- test[,-1]

# run regression on training data
train.reg <- lm(salespr ~ bedrooms + acres + age + totbaths + 
          sqftheat + squnheat, data=train) 
summary(train.reg)     

# now calculate the predictions on the test data
test.preds <- as.vector(predict(object=train.reg, newdata=test.X))
test.preds

# compute training and test sample R-squared value
library(miscTools)
train.r2 <- rSquared(y=train[,1], resid=resid(train.reg))
test.r2  <- rSquared(y=test.Y, resid= (test.Y - test.preds))

round(train.r2,4)
round(test.r2,4)

# the dangers of overfitting
# let's create 100 new variables, that are randomly generated data from a normal distribution
# these variables should have NO EFFECT on the sales prices of homes
set.seed(1234567)
X.random <- matrix(rnorm(100*N), nrow=N, ncol=100)
colnames(X.random) <- paste("X", 1:100, sep="")
X.random[1:5,] # first 5 rows

#combine old data with new fake data
my.new.data.set <- cbind.data.frame(my.data.set, X.random)

# let's recreate new training and test datasets that include these 100 new variables
train.new <- my.new.data.set[1:K, ]
train.Y.new <- train.new[,1]  
train.X.new <- as.matrix(train.new[,-1])  

test.new  <- my.new.data.set[(K+1):N, ]
test.Y.new <- test.new[,1]
test.X.new <- test.new[,-1]

# let's rerun our regression using the old variables + the new fake variables
train.reg.new <- lm(salespr ~ ., data = train.new) 
summary(train.reg.new)
summary(train.reg.new)$r.squared # extract r-squared value from training data regression

# some p-values for the randomly generated data look "significant"
# but this data is random and has nothing to do with the price of homes
sort(round(coef(summary(train.reg.new))[-(1:7),4],4))

# calculate out of sample predictions for test new data
test.preds.new <- as.vector(predict(object=train.reg.new, newdata=test.X.new))

# calculate r2 for both training new and test new data
train.r2.new <- round(rSquared(y=train.Y.new, resid=resid(train.reg.new)),4)
test.r2.new <- round(rSquared(y=test.Y.new, resid= (test.Y.new - test.preds.new)),4)
c(train.r2.new, test.r2.new) #display r2 values for train and test samples

# multicollinearity illustration
sales3 <- read.table("D:/data/SALES3.txt", header=T, sep="\t")
attach(sales3)
sales3 # look at it

# examine correlations for sales3 dataset
round(cor(sales3),4) # it looks like Trad and Int are highly correlated

# regression of Sales on Int. Looks like Int matters
summary(lm(Sales ~ Int))

# regression of Sales on Trad. Looks like Trad matters too
summary(lm(Sales ~ Trad))

# now also include Trad and Int together. Looks like neither variable matters now!
summary(lm(Sales ~ Int + Trad))

# combine Int and Trad
Total <- Int + Trad
summary(lm(Sales ~ Total))

detach(sales3)
rm(list=ls(all=T))

## Forward and Backward Selection
# let's use variable selection on the real estate data
homes3 <- read.table("D:/data/real estate data part two.txt", header=T, sep="\t")
attach(homes3)

# let's create a matrix with a bunch of potential variables
X <- cbind(bedrooms, acres, age,
totbaths, numgarge, numfirep, daysmkt,  closespr, closesum,
closefal, basement, diningrm, familyrm, office, 
patio, deck, porch, stycontm, styranch, stytradt,
stytrans, brick, cedar, fiber, hardwood, vinyl,
elecheat, elemdist, middist, highdist, onestory, sqftheat, squnheat)

# create a data frame that has Y and your X variables all in one place
new.home.data <- cbind.data.frame(salespr, X)

# forward selection
# first fit a model with no variables (only the intercept)
null.model <- lm(salespr ~ 1) # intercept only model
summary(null.model)

# a regression model with only an intercept predicts that each home will
# sell for the average price of all homes in your data
# an intercept only model predicts the mean selling price for all homes
as.vector(fitted(null.model))
mean(salespr)

# now fit the full model with all X variables
full.model <- lm(salespr ~ ., data = new.home.data)
summary(full.model) # it looks like some of the variables don't matter

# now perform a forward variable selection regression model 
# it looks like 20 out of the 33 variables are retained
forward.results <- step(object=null.model, direction="forward", scope=formula(full.model))
summary(forward.results)

# backward elimination
backward.results <- step(object=full.model, direction="backward")
summary(backward.results)

# which variables are selected in forward selection?
forward.vars.used <- as.data.frame(sort(names(coef(forward.results)[-1])))
colnames(forward.vars.used) <- "Forward"
forward.vars.used <- cbind.data.frame(forward.vars.used, "F")
forward.vars.used

# which variables are selected in backward selection?
backward.vars.used <- as.data.frame(sort(names(coef(backward.results)[-1])))
colnames(backward.vars.used) <- "Backward"
backward.vars.used <- cbind.data.frame(backward.vars.used, "B")
backward.vars.used

# which variables are in the forward model and not in backward model, and vice-versa
vars.selected <- merge(x=forward.vars.used, y=backward.vars.used, by.x="Forward", by.y="Backward", all=TRUE)
vars.selected # 2 vars in backward but not forward. 1 var in forward but not backward

## Performing Log transformations to Reduce Skewness of Data
# transformations of X
Y <- c(5, 9, 14.5, 13.3, 16, 17.4, 18, 17.5, 18.8, 18)
X <- 1:10

# log of x
logX <- log(X)

# X squared
X2 <- X^2

# basic regression
summary(lm(Y ~ X))

# log transformed regression
summary(lm(Y ~ logX))

# polynomial regression
summary(lm(Y ~ X + X2)) # include both X and X2 in the model

## Ordered Probit 
# what about when more than two possible outcomes are possible for Y?
# let's look at a quick example
# run an ordered probit
data = read.table(file="D:/data/ordered probit data.txt", header=T, sep="\t")
newdata = read.table(file="D:/data/ordered probit newdata.txt", header=T, sep="\t")

# data has survey results of people's willingness to pay money to maintain national parks
# DV is WTP. Values are:
# 1 = "not willing to pay at all"
# 2 = "willing to pay a little bit"
# 3 = "willing to pay a moderate amount"
# 4 = "willing to pay a lot of money"

# this is an ordered categorical model
data[1:10,]
newdata

# variables used to predict willingness to pay are age, gender, and income
library(MASS)

# let's run an ordered probit model to model the probability of willingness to pay 1, 2, 3, or 4
attach(data)

# we must convert dependent variable WTP into an ordered factor variable so R knows 1,2,3,4
# are ordered categories and not the actual numbers 1,2,3,4
WTP <- factor(as.numeric(WTP), ordered=TRUE, levels=c("1","2","3","4"))

ordered.model <- polr(WTP ~ age + female + income,  method="probit")
summary(ordered.model)

# the model spits out the probability of of each person being in each of the four categories
fits <- fitted.values(ordered.model)
fits[1:10,] # first 10 people

# the rows all sum to 1.00 as the probabilities must sum to 1.
apply(fits,1,sum)

# calculate the most likely group each person is in based on maximum probability
maxes <- apply(fits,1,max)
predicted.group <- sapply(1:nrow(fits),function(i){match(maxes[i], fits[i,])})
cbind(fits[61:70,], predicted.group[61:70])

# distribution of groups. no one is predicted to be in the "low" payment group of 2
table(predicted.group)

# dataset newdata contains 20 new people who were not surveyed.
# based on their age, gender, and income we want to predict how much they are willing to pay
newdata[1:10,] # no WTP variable

# write a function to predict the new people's probabilities of being in each group
predict.ordered.probit <- function(model, data.matrix){
     
     # extract model coefficients and cutpoints
     coefs <- model[[1]]
     cutpoints <- model[[2]]
     
     # compute linear predictor
     linpred <- as.vector(data.matrix%*%coefs)
     
     # number of cutpoints
     K <- length(cutpoints)
     
     # calculate predicted probabilities
     predprobs <- NULL
     for(k in 1:(K+1)){
          if(k == 1){
               temp <- pnorm(q = cutpoints[k] - linpred, mean=0, sd=1)			
               predprobs <- cbind(predprobs, temp)
          } else { 
               if(k <= K){
                    temp <- pnorm(q = cutpoints[k] - linpred, mean=0, sd=1) - pnorm(q = cutpoints[k-1] - linpred, mean=0, sd=1)			
                    predprobs <- cbind(predprobs, temp)
               } else { 
                    temp <- 1 - pnorm(q = cutpoints[k-1] - linpred, mean=0, sd=1)			
                    predprobs <- cbind(predprobs, temp)
               }
          }
     }
     colnames(predprobs) <- 1:(K+1)
     return(predprobs)
}

X <- as.matrix(newdata) # convert data frame into a data matrix

# extract new predicted probabilities
newfits <- predict.ordered.probit(model = ordered.model, data.matrix = X)
newfits

# we can again predict the most likely group by taking the column with the largest probability
newmaxes <- apply(newfits,1,max)
new.predicted.group <- sapply(1:nrow(newfits),function(i){match(newmaxes[i], newfits[i,])})
cbind(newfits,new.predicted.group)
table(new.predicted.group)
