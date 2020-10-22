Online News Analysis - Predicting Media Shares by Article
Characteristics
================
Matt Kasle
10/15/2020

# tuesday articles

## Introduction

This dataset summarizes a heterogeneous set of features about articles
published by Mashable in a period of two years. The goal is to predict
the number of shares in social networks (popularity).

There are 61 attributes, 58 which are predictive attributes, 2 of which
are non-predictive (url and timestamp), and 1 that is the target.

The types of veriables include the number of words in the article, the
number of words in the title, the positivity and sentimentality of the
article, the article’s subject, the number of keywords used, and much
more.

The objective of this project is to predict the number of social media
shares using two different tree-based algorithms. The first algorithm
will be a non-ensemble regression tree, and the second algorithm will be
a boosted trees algorithm, which is a state-of-the-art classification
technique.

The other objective of this project is to create an automated report
that outputs the classification analysis for each articles published on
each weekday.

The required packages to run this analysis are tidyverse, caret, tree,
and patchwork.

## Data

First, we need to read in the data and set up the subsetting of the data
into a single weekday. Then, we’ll split the data set to a training and
test set for training and evaluating the classificaiton models. A 70/30
train-test split will be used.

``` r
set.seed(1)
library(tidyverse)
library(caret)
library(tree)

news <- read_csv("OnlineNewsPopularity.csv")
```

``` r
varDayOfWeek <- parse(text=paste0("weekday_is_", params$day_of_week))
news <- news %>% filter(eval(varDayOfWeek) == 1)

news <- news %>% select(-url, -timedelta, -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday,
                        -weekday_is_thursday,-weekday_is_friday, 
                        -weekday_is_saturday, -weekday_is_sunday, -is_weekend)

newsIndex <- createDataPartition(news$shares, p = 0.3, list = FALSE)
newsTrain <- news[newsIndex, ]
newsTest <- news[-newsIndex, ]
```

## Summarizations

Below are summary statistics of the online news data set, such as the
number of observations in the train set, a numerical summary of the
response variable (number of social media shares), a distribution of the
response, and relationships between the response and interesting
variables in the dataset.

Number of rows in training set:

``` r
nrow(newsTrain)
```

    ## [1] 2218

Summary of response variable:

``` r
summary(newsTrain$shares)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##     44.0    897.2   1300.0   2976.0   2500.0 145500.0

It is important to make note of the minimum and maximum of the response
variable, as well as the median and quartiles. For future analysis, it
may be best to remove outliers.

Distribution of response variable:

``` r
ggplot(data = newsTrain, aes(x = shares)) +
  geom_histogram() + xlab("Shares") + ggtitle(paste("Distribution of Shares in Training Data - ",
                                                    params$day_of_week))
```

![](tuesday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Below is a plot of four interesting variables to the response variable,
shares. We want to see if there are any intersting patterns here.

``` r
library(patchwork)
par(mfrow=c(2,2))
plot1 <- ggplot(data=news, aes(x = news$n_unique_tokens, y = shares)) + geom_point(stat = "identity") +
    labs(x="Unique Words in Article", y="Shares")

plot2 <- ggplot(data=news, aes(x = news$rate_positive_words, y = shares)) + geom_point(stat = "identity") +
    labs(x="Rate of Positive Words", y="")

plot3 <- ggplot(data=news, aes(x = news$rate_negative_words, y = shares)) + geom_point(stat = "identity") +
    labs(x="Rate of Negative Words", y="Shares")

plot4 <- ggplot(data=news, aes(x = news$global_sentiment_polarity, y = shares)) + geom_point(stat = "identity") +
    labs(x="Sentiment Polarity", y="")

plot1 + plot2 + plot3 + plot4
```

    ## Warning: Use of `news$n_unique_tokens` is discouraged. Use `n_unique_tokens` instead.

    ## Warning: Use of `news$rate_positive_words` is discouraged. Use `rate_positive_words` instead.

    ## Warning: Use of `news$rate_negative_words` is discouraged. Use `rate_negative_words` instead.

    ## Warning: Use of `news$global_sentiment_polarity` is discouraged. Use `global_sentiment_polarity` instead.

![](tuesday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

There are six different article subjects. It would be interesting to
know if some subjects are more popular than others, so we can plot the
median shares by the different article types.

``` r
news['article_type'] <- ifelse(news$data_channel_is_lifestyle == 1, "Lifestyle",
       ifelse(news$data_channel_is_entertainment == 1, "Entertainment",
       ifelse(news$data_channel_is_world == 1, "World",
       ifelse(news$data_channel_is_bus == 1, "Bus",
       ifelse(news$data_channel_is_socmed == 1, "SocMed",
       ifelse(news$data_channel_is_tech == 1, "Tech","Other"))))))

ggplot(data=news, aes(x = article_type, y = shares)) + geom_bar(stat = "summary", fun.y = "median") +
    labs(x="Article Type", y="Median Shares", title="Median Shares by Article Type")
```

    ## Warning: Ignoring unknown parameters: fun.y

    ## No summary function supplied, defaulting to `mean_se()`

![](tuesday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
# drop column used for plotting
news <- news %>% select(-article_type)
```

## Modeling

There should be text describing the type of model you are fitting, your
fitting process, and the final chosen model (this last part is to be
automated so I don’t expect you to explicitly interpret that model, but
you should be able to display something about the final model chosen on
the training data).

### Regression Tree Model

The first model fit to the data will be a regression tree. We’ll use
leave-one-out cross-validation to determine the optimal size of the
model, as defined by number of splits. By plotting the fitted tree, we
can see the deviance by tree size (larger deviance means a better fit).

``` r
treeFit <- tree(shares ~ ., data = newsTrain)
summary(treeFit)
```

    ## 
    ## Regression tree:
    ## tree(formula = shares ~ ., data = newsTrain)
    ## Variables actually used in tree construction:
    ## [1] "self_reference_min_shares"  "avg_negative_polarity"      "kw_max_avg"                 "global_rate_negative_words"
    ## [5] "n_non_stop_unique_tokens"  
    ## Number of terminal nodes:  6 
    ## Residual mean deviance:  40830000 = 9.031e+10 / 2212 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -30780.00  -2074.00  -1057.00      0.00    -73.89 115300.00

``` r
pruneFit <- cv.tree(treeFit,
                  K=nrow(newsTrain)-1
                  )


pruneFitFinal <- prune.tree(treeFit, best = pruneFit$size[1]) 

plot(pruneFit$size ,pruneFit$dev ,type="b")
```

![](tuesday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

### Boosted tree model

Next, we’ll fit a boosted regression tree. The boosted tree algorithm
has a few hyperparameters, and we’ll use repeated 10-fold
cross-validation to determine the optimal hyperparameter values. The
hyperparameters of the optimal boosted tree is printed below, as well a
summary of each fitted model.

``` r
fitControl <- trainControl(## 10-fold CV
                           method = "repeatedcv",
                           number = 10,
                           ## repeated ten times
                           repeats = 5
                           )

boostedFit <- train(shares ~ ., data = newsTrain, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE
                 )

boostedFit$bestTune
```

    ##   n.trees interaction.depth shrinkage n.minobsinnode
    ## 1      50                 1       0.1             10

``` r
boostedFit$results
```

    ##   shrinkage interaction.depth n.minobsinnode n.trees     RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1       0.1                 1             10      50 6381.309 0.02928061 2661.221 2502.505 0.04219204 382.7203
    ## 4       0.1                 2             10      50 6411.720 0.03325144 2663.336 2463.552 0.03909676 377.4254
    ## 7       0.1                 3             10      50 6388.921 0.03883563 2675.423 2448.070 0.04882380 375.5190
    ## 2       0.1                 1             10     100 6445.416 0.02945881 2699.439 2469.540 0.03498977 393.7136
    ## 5       0.1                 2             10     100 6504.834 0.03201192 2717.170 2412.158 0.04098878 382.6605
    ## 8       0.1                 3             10     100 6476.024 0.03456451 2727.204 2400.207 0.04320664 381.8427
    ## 3       0.1                 1             10     150 6477.886 0.03036885 2723.684 2448.382 0.03592404 385.7473
    ## 6       0.1                 2             10     150 6536.299 0.03360968 2733.990 2395.384 0.04484307 379.1525
    ## 9       0.1                 3             10     150 6526.301 0.03517714 2770.555 2363.523 0.04366623 376.9903

### Linear regression model

Next, we’ll fit a multiple linear regression model on train data.
Summary of the model is printed below.

``` r
linearfit<-lm(shares~.,data=newsTrain)
summary(linearfit)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = newsTrain)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -13668  -2080   -857    424 130831 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   -2.013e+03  2.190e+03  -0.919 0.358097    
    ## n_tokens_title                 1.625e+02  7.023e+01   2.314 0.020780 *  
    ## n_tokens_content               3.279e-01  5.210e-01   0.629 0.529146    
    ## n_unique_tokens                1.275e+01  4.758e+03   0.003 0.997862    
    ## n_non_stop_words              -1.836e+03  7.377e+03  -0.249 0.803440    
    ## n_non_stop_unique_tokens       1.675e+03  3.938e+03   0.425 0.670687    
    ## num_hrefs                      1.566e+00  1.317e+01   0.119 0.905348    
    ## num_self_hrefs                -9.412e+01  5.028e+01  -1.872 0.061319 .  
    ## num_imgs                       6.012e+01  2.068e+01   2.907 0.003688 ** 
    ## num_videos                     4.693e+01  3.623e+01   1.295 0.195341    
    ## average_token_length          -2.562e+02  5.976e+02  -0.429 0.668184    
    ## num_keywords                  -7.385e+01  9.194e+01  -0.803 0.421927    
    ## data_channel_is_lifestyle     -5.982e+02  1.052e+03  -0.569 0.569753    
    ## data_channel_is_entertainment -9.334e+02  6.221e+02  -1.500 0.133664    
    ## data_channel_is_bus           -5.069e+02  9.305e+02  -0.545 0.585979    
    ## data_channel_is_socmed        -7.339e+01  9.064e+02  -0.081 0.935475    
    ## data_channel_is_tech          -4.785e+01  9.441e+02  -0.051 0.959584    
    ## data_channel_is_world         -1.487e+02  9.356e+02  -0.159 0.873722    
    ## kw_min_min                     3.256e+00  4.075e+00   0.799 0.424386    
    ## kw_max_min                    -1.147e-01  1.920e-01  -0.598 0.550215    
    ## kw_avg_min                     4.369e-01  1.118e+00   0.391 0.695976    
    ## kw_min_max                    -1.474e-03  2.977e-03  -0.495 0.620600    
    ## kw_max_max                     1.062e-03  1.478e-03   0.719 0.472367    
    ## kw_avg_max                    -1.865e-03  2.043e-03  -0.913 0.361344    
    ## kw_min_avg                    -6.252e-01  1.883e-01  -3.320 0.000915 ***
    ## kw_max_avg                    -2.805e-01  6.283e-02  -4.465 8.42e-06 ***
    ## kw_avg_avg                     1.696e+00  3.728e-01   4.549 5.69e-06 ***
    ## self_reference_min_shares      8.199e-02  3.488e-02   2.351 0.018826 *  
    ## self_reference_max_shares     -4.690e-03  1.812e-02  -0.259 0.795767    
    ## self_reference_avg_sharess     1.475e-02  4.888e-02   0.302 0.762954    
    ## LDA_00                         5.387e+02  1.097e+03   0.491 0.623404    
    ## LDA_01                        -4.853e+02  1.273e+03  -0.381 0.703030    
    ## LDA_02                        -7.511e+02  1.099e+03  -0.683 0.494531    
    ## LDA_03                        -3.796e+02  1.220e+03  -0.311 0.755681    
    ## LDA_04                                NA         NA      NA       NA    
    ## global_subjectivity            4.179e+03  2.089e+03   2.001 0.045544 *  
    ## global_sentiment_polarity     -6.303e+03  4.139e+03  -1.523 0.127956    
    ## global_rate_positive_words    -8.199e+03  1.844e+04  -0.445 0.656586    
    ## global_rate_negative_words    -1.845e+04  3.594e+04  -0.513 0.607742    
    ## rate_positive_words           -7.882e+02  6.845e+03  -0.115 0.908343    
    ## rate_negative_words           -4.210e+01  7.119e+03  -0.006 0.995282    
    ## avg_positive_polarity         -2.475e+03  3.410e+03  -0.726 0.468041    
    ## min_positive_polarity          5.520e+03  2.937e+03   1.880 0.060293 .  
    ## max_positive_polarity          1.603e+03  1.047e+03   1.531 0.125957    
    ## avg_negative_polarity          1.303e+03  3.074e+03   0.424 0.671699    
    ## min_negative_polarity         -1.321e+03  1.091e+03  -1.212 0.225748    
    ## max_negative_polarity         -4.873e+03  2.554e+03  -1.908 0.056551 .  
    ## title_subjectivity            -5.273e+02  6.436e+02  -0.819 0.412654    
    ## title_sentiment_polarity       1.476e+03  5.859e+02   2.520 0.011811 *  
    ## abs_title_subjectivity         8.047e+02  8.854e+02   0.909 0.363529    
    ## abs_title_sentiment_polarity   1.421e+03  9.345e+02   1.520 0.128614    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6586 on 2168 degrees of freedom
    ## Multiple R-squared:  0.09608,    Adjusted R-squared:  0.07565 
    ## F-statistic: 4.703 on 49 and 2168 DF,  p-value: < 2.2e-16

## Model Evaluations

Finally, we’ll evaluate the performance of each model by seeking the
lowest root mean squared error of its predictions on the test dataset
when compared to the actual values in the dataset. This should be a good
approximation of the model’s performance on unseen data.

### Regression Tree

Below is the RMSE of the optimal (non-ensemble) regression tree:

``` r
treePred <- predict(pruneFitFinal, newdata = dplyr::select(newsTest, -shares))
sqrt(mean((treePred-newsTest$shares)^2))
```

    ## [1] 10923.96

### Boosted Trees

Below is the RMSE of the optimal boosted regression tree:

``` r
boostedTreePred <- predict(boostedFit, newdata = dplyr::select(newsTest, -shares))
sqrt(mean((boostedTreePred-newsTest$shares)^2))
```

    ## [1] 10822.84

### linear regression

Below is the RMSE of the multiple linear model:

``` r
linearPred<-predict(linearfit, newdata=dplyr::select(newsTest, -shares))
sqrt(mean((linearPred-newsTest$shares)^2))
```

    ## [1] 15742.26
