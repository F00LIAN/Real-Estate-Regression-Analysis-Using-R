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
- Required R libraries: `readxl`.
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
homes2 <- as.data.frame(read_excel("D:/homes2.xlsx", sheet="Sheet1"))
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
homes3 <- read.table("real estate data.txt", header=T, sep="\t")
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
new.homes <- read.table("20 new homes.txt", header=T, sep="\t")
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

### 16. Load in New Data
```r
# run a new regression model
homes3 <- read.table("D:/data/real estate data.txt", header=T, sep="\t")
attach(homes3)

# suppress scientific notation
options(scipen=999)
par(mfrow=c(1,1))
```
<p> <img src="https://github.com/user-attachments/assets/3bd08e2b-46cf-43fc-a06d-785a52a0975c" height="60%" width="60%" alt="Regression Summary"/> </p>









<h2>Conclusion</h2>

- Simple linear regression was used to model house prices based on floor area.
- Multiple regression analysis improved predictive power using multiple features.
- Dummy variables accounted for seasonal variations in sales price.
- Model diagnostics such as R² and RMSE were used to evaluate performance.

<h2>Future Improvements</h2>

- Experiment with log transformations to handle non-linearity.
- Implement ridge and lasso regression for better feature selection.
- Use decision tree regression for non-linear relationships.



