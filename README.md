<p align="center">
<img src="https://github.com/user-attachments/assets/3dc845f1-74a1-40ab-aaa6-c7980e6a41f2" height="40%" width="50%" alt="Regression Analysis"/>
</p>

<h1>Regression Analysis in R</h1>
This project demonstrates linear and multiple regression analysis using R. It explores real estate pricing models, regression diagnostics, and the use of dummy variables to account for categorical data.<br />

<h2>Video Demonstration</h2>

- ### [YouTube: Regression Analysis in R](https://www.youtube.com/watch?v=wNsKf7wSqhQ)

<h2>Environments and Technologies Used</h2>

- R Programming
- RStudio
- Data Analysis and Regression Modeling

<h2>Operating Systems Used </h2>

- Windows 11

<h2>List of Prerequisites</h2>

Before running this project, ensure you have:
- R and RStudio installed.
- Required R libraries: `readxl`, `miscTools`.
- Access to datasets (`homes2.xlsx`, `real estate data.txt`, `20 new homes.txt`).

<h2>Installation Steps</h2>

### 1. Install Required Libraries
```r
install.packages("readxl")
```

### 2. Load and Preprocess Data
```r
library("readxl")

# Load Housing Data
homes2 <- as.data.frame(read_excel("D:/data/homes2.xlsx", sheet="Sheet1"))
attach(homes2)
```

<h2>Simple Linear Regression</h2>

### 3. Visualizing Data
```r
plot(Floor, Price, frame=F, pch=19, cex=2, font.axis=2, font.lab=2)
```
<p> <img src="https://github.com/user-attachments/assets/9b812c47-0b98-4842-b11e-ed9a393bfad8" height="30%" width="30%" alt="Scatter Plot with Regression"/> </p>

### 4. Fitting a Simple Linear Regression Model
```r
reg1 <- lm(Price ~ Floor)
summary(reg1)
```
<p> <img src="https://github.com/user-attachments/assets/ece49880-2b66-4099-a5b6-56c731290dba" height="60%" width="60%" alt="Regression Line"/> </p>

### 5. Adding Regression Line
```r
abline(reg1, col="red", lwd=2)
```
<p> <img src="https://github.com/user-attachments/assets/5c168eba-5058-4474-ac86-dfcc6b55aa1c" height="30%" width="30%" alt="Regression Line"/> </p>

<h2>Regression Diagnostics</h2>

### 6. Summary Results and Extracting Fitted Values and Residuals
```r
# fit statistics and model results
summary(reg1)

fits <- fitted(reg1)
resid <- resid(reg1)
```

<p> <img src="https://github.com/user-attachments/assets/7a3b3d47-1822-49b2-b082-7ec773c00c5f" height="60%" width="60%" alt="Summary"/> </p>

### 7. Overlaying Fitted Values
```r
points(Floor, fits, pch=4, col="darkblue", lwd=2)
```
<p> <img src="https://github.com/user-attachments/assets/553f2547-4dc4-4472-9683-0fac3e955bea" height="30%" width="30%" alt="Fitted Values"/> </p>

<h2>Multiple Regression Analysis</h2>

### 8. Load the Large Real Estate Dataset and Run Test Regression
```r
homes3 <- read.table("D:/data/real estate data.txt", header=T, sep="\t")
attach(homes3)

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
```

<p> <img src="https://github.com/user-attachments/assets/4154d295-864d-46e6-8d79-071737e75713" height="30%" width="30%" alt="Plot"/> </p>

- The Expected Selling Price of a house that has 2000 heated square feet is $266,690

### 9. Fit Multiple Regression Model
```r
reg3 <- lm(salespr ~ bedrooms + acres + age + totbaths + sqftheat + squnheat)
summary(reg3)
```

### 10. Predicting Prices for New Homes
```r
new.homes <- read.table("D:/data/20 new homes.txt", header=T, sep="\t")
sales.prices <- predict(object=reg3, newdata=new.homes)

# Combine Predicted Prices with Data
sales20 <- cbind(new.homes, sales.prices)
```
<p> <img src="https://github.com/user-attachments/assets/41f68a80-9869-40c5-b5bc-bc94ed33a169" height="60%" width="60%" alt="Predicted Prices"/> </p>

<h2>Using Dummy Variables in Regression</h2>

### 11. Regression with Seasonal Dummy Variables
```r
reg4 <- lm(salespr ~ sqftheat + closesum)
summary(reg4)
``` 

### 12. Plot Regression with Dummy Variables
```r
plot(sqftheat, salespr, frame=F, pch=19, cex=.5, font.axis=2, font.lab=2,
     xlab="Square Feet (Heated)", ylab="Sales Price", xlim=c(1000,3000),
     ylim=c(100000,500000))

# Calculate predictions for summer vs. non-summer sales
coefs3 <- coef(reg4)
sales.no.summer <- coefs3[1] + coefs3[2]*sqftheat
sales.summer    <- coefs3[1] + coefs3[2]*sqftheat + coefs3[3]

# Plot results
lines(sqftheat, sales.no.summer, lwd=2, col="blue") # Homes not sold in summer
lines(sqftheat, sales.summer, lwd=2, col="darkgreen") # Homes sold in summer
```
<p> <img src="https://github.com/user-attachments/assets/44ad7ad9-78ef-476c-83fc-99922adcc52a" height="40%" width="40%" alt="Dummy Variable Regression"/> </p>

### 13. Changing Reference Group
```r
close_all_other <- 1 - closesum
reg5 <- lm(salespr ~ sqftheat + close_all_other)
summary(reg5)
```

<h2>Evaluate Model Performance</h2>

### 14. Extracting Root Mean Squared Error (RMSE)
```r
sigma(reg7)
```

### 15. Calculate RMSE Manually
```r
errors.2 <- resid(reg7)^2
N <- length(salespr)
K <- length(coef(reg7))
sqrt(sum(errors.2)/(N - K))
```

### 16. Checking R-Squared Value
```r
summary(reg7)
```
<p> <img src="https://github.com/user-attachments/assets/3bd08e2b-46cf-43fc-a06d-785a52a0975c" height="60%" width="60%" alt="Regression Summary"/> </p>

<h2>Advanced Regression Techniques</h2>

### 16. Load in New Data and Run New Regression Model
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/9cbe63fa-8db7-40bd-a165-7e19778e6c4c" height="60%" width="60%" alt="Regression Summary"/> </p>

### 17. Identifying Influential Observation with Cook's Distance
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/5e6b8609-58b7-41c7-8dcf-b148c1275098" height="60%" width="60%" alt="Cooks Distance"/> </p>

- Cook's distance helps identify observations with undue influence on model coefficients.

- Observations above a threshold might be considered outliers or high leverage points.

### 18. Evaluating Model Functionality on Unseen Test Data
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/b8e1e163-728b-4123-a47c-7853e45f271d" height="60%" width="60%" alt="Train-Test Eval"/></p>

- Splitting data into training/test partitions for an unbiased measure of out-of-sample predictive performance.

- Compare training R-squared vs. test R-squared to assess overfitting.

### 19. Assessing the Dangers of Overfitting Data
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/424fa438-4693-4bcf-9141-53d22e48552d" height="60%" width="60%" alt="Overfitting"/> </p>

- Adding random predictors can inflate in-sample R-squared scores but this typically reduces out-of-sample performance.

- Significant p-values may sometimes appear by chance with many predictors. 

### 20. Multicollinearity Illustration 
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/e8ad359e-ac9f-4d27-a733-12018d5a86fa" height="60%" width="60%" alt="Multicollinearity"/> </p>

- When predictors are highly correlated, it can mask and inflate their true predictive power.

- Sometimes it makes sense to summarize and combine correlated features into a single composite column to alleviate the issue.

### 21. Forward and Backward Selection
```r
### let's use variable selection on the real estate data
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
```
<p> <img src="https://github.com/user-attachments/assets/4ee0f40c-a0f4-493d-aad0-fda4edffba51" height="60%" width="60%" alt="Difference Between Forward and Backward Selection"/> </p>

- Forward selection starts with no predictors, adding them one by one.
  
- Backward elimination starts with all predictors and removes them step by step.

- Stepwise methods can yield different subsets, so results should be interpreted carefully.

### 21. Performing Log Transformations to Reduce Skewness of Data
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/3d6082b5-936c-407d-aecd-f5958fc7218f" height="60%" width="60%" alt="Log Transformation"/> </p>

- Log transformations or polynomial terms can capture nonlinear relationships that a standard linear model might miss.

- Always interpret transformed coefficients carefully (e.g., logs interpret as percentage changes).

### 22. Ordered Probit
```r
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
```
<p> <img src="https://github.com/user-attachments/assets/d278aca1-c766-488f-adc2-d3f2a12a4a65" height="60%" width="60%" alt="Predicting Outcomes"/> </p>

- Ordered probit/logit is used when the dependent variable has more than two levels with a natural ordering.

- Fitted probabilities can be used to classify observations into the most likely category.
  
<h2>Conclusion</h2>

- Simple linear regression was used to model house prices based on floor area.
- Multiple regression analysis improved predictive power using multiple features.
- Dummy variables accounted for seasonal variations in sales price.
- Model diagnostics such as R² and RMSE were used to evaluate performance.

<h2>Future Improvements</h2>

- Experiment with log transformations to handle non-linearity.
- Implement ridge and lasso regression for better feature selection.
- Use decision tree regression for non-linear relationships.



