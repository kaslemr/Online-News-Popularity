Online News Analysis - Predicting Media Shares by Article
Characteristics
================
Matt Kasle
10/15/2020

# saturday articles

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

    ## [1] 738

Summary of response variable:

``` r
summary(newsTrain$shares)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      52    1300    2000    4878    3600  617900

It is important to make note of the minimum and maximum of the response
variable, as well as the median and quartiles. For future analysis, it
may be best to remove outliers.

Distribution of response variable:

``` r
ggplot(data = newsTrain, aes(x = shares)) +
  geom_histogram() + xlab("Shares") + ggtitle(paste("Distribution of Shares in Training Data - ",
                                                    params$day_of_week))
```

![](saturday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](saturday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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

![](saturday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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
    ## [1] "kw_max_max" "kw_avg_max"
    ## Number of terminal nodes:  3 
    ## Residual mean deviance:  428100000 = 3.147e+11 / 735 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -152600.0   -2556.0   -1856.0       0.0    -256.4  463800.0

``` r
pruneFit <- cv.tree(treeFit,
                  K=nrow(newsTrain)-1
                  )


pruneFitFinal <- prune.tree(treeFit, best = pruneFit$size[1]) 

plot(pruneFit$size ,pruneFit$dev ,type="b")
```

![](saturday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
    ## 1       0.1                 1             10      50 14821.14 0.03444991 5325.925 19490.11 0.06086296 2368.729
    ## 4       0.1                 2             10      50 15623.30 0.02622291 5584.830 19222.25 0.04492167 2399.516
    ## 7       0.1                 3             10      50 15656.33 0.02784397 5532.500 19163.95 0.05774164 2314.904
    ## 2       0.1                 1             10     100 15429.96 0.02974613 5886.818 19256.73 0.04870798 2205.154
    ## 5       0.1                 2             10     100 16679.37 0.02687581 6355.615 18834.58 0.04326404 2206.836
    ## 8       0.1                 3             10     100 16689.68 0.02342865 6271.445 18821.08 0.05219615 2189.030
    ## 3       0.1                 1             10     150 15890.49 0.02910195 6212.299 19112.69 0.04713341 2167.935
    ## 6       0.1                 2             10     150 17315.52 0.02419487 6879.455 18620.36 0.03935443 2112.950
    ## 9       0.1                 3             10     150 17346.63 0.02138460 6740.321 18583.79 0.05474449 2060.298

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
    ## -22670  -4813   -992   2579 581899 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)  
    ## (Intercept)                    4.511e+03  1.369e+04   0.329   0.7419  
    ## n_tokens_title                 4.106e+02  4.725e+02   0.869   0.3852  
    ## n_tokens_content              -3.819e+00  3.969e+00  -0.962   0.3362  
    ## n_unique_tokens               -3.347e+04  3.330e+04  -1.005   0.3153  
    ## n_non_stop_words               5.196e+03  2.782e+04   0.187   0.8519  
    ## n_non_stop_unique_tokens       9.528e+03  2.776e+04   0.343   0.7315  
    ## num_hrefs                      4.055e+01  1.053e+02   0.385   0.7002  
    ## num_self_hrefs                -7.810e+01  2.374e+02  -0.329   0.7423  
    ## num_imgs                       2.912e+01  1.469e+02   0.198   0.8429  
    ## num_videos                    -1.340e+02  3.332e+02  -0.402   0.6876  
    ## average_token_length           6.331e+02  4.141e+03   0.153   0.8785  
    ## num_keywords                   6.450e+01  5.864e+02   0.110   0.9124  
    ## data_channel_is_lifestyle     -7.997e+03  5.259e+03  -1.521   0.1288  
    ## data_channel_is_entertainment -4.810e+03  3.995e+03  -1.204   0.2290  
    ## data_channel_is_bus           -7.761e+03  6.130e+03  -1.266   0.2059  
    ## data_channel_is_socmed        -1.106e+04  5.482e+03  -2.018   0.0440 *
    ## data_channel_is_tech          -9.459e+03  5.624e+03  -1.682   0.0930 .
    ## data_channel_is_world         -1.438e+04  5.583e+03  -2.576   0.0102 *
    ## kw_min_min                     4.705e+01  2.595e+01   1.813   0.0703 .
    ## kw_max_min                     6.145e-01  1.973e+00   0.312   0.7555  
    ## kw_avg_min                    -4.500e+00  1.168e+01  -0.385   0.7001  
    ## kw_min_max                     1.389e-02  1.756e-02   0.791   0.4294  
    ## kw_max_max                    -2.476e-03  9.806e-03  -0.252   0.8008  
    ## kw_avg_max                    -1.616e-02  1.414e-02  -1.143   0.2533  
    ## kw_min_avg                     5.647e-01  1.179e+00   0.479   0.6322  
    ## kw_max_avg                    -1.058e-01  2.959e-01  -0.358   0.7208  
    ## kw_avg_avg                     1.167e+00  2.308e+00   0.505   0.6134  
    ## self_reference_min_shares     -1.265e-01  2.054e-01  -0.616   0.5380  
    ## self_reference_max_shares     -9.881e-02  1.702e-01  -0.581   0.5616  
    ## self_reference_avg_sharess     2.392e-01  3.662e-01   0.653   0.5138  
    ## LDA_00                         1.807e+03  7.346e+03   0.246   0.8058  
    ## LDA_01                        -7.136e+03  8.120e+03  -0.879   0.3798  
    ## LDA_02                         7.903e+03  7.304e+03   1.082   0.2796  
    ## LDA_03                         1.650e+03  7.703e+03   0.214   0.8305  
    ## LDA_04                                NA         NA      NA       NA  
    ## global_subjectivity           -1.373e+04  1.370e+04  -1.002   0.3165  
    ## global_sentiment_polarity     -2.174e+03  2.904e+04  -0.075   0.9403  
    ## global_rate_positive_words    -1.020e+05  1.194e+05  -0.854   0.3934  
    ## global_rate_negative_words     3.498e+05  2.541e+05   1.376   0.1692  
    ## rate_positive_words            2.604e+04  2.035e+04   1.280   0.2011  
    ## rate_negative_words                   NA         NA      NA       NA  
    ## avg_positive_polarity         -7.329e+03  2.338e+04  -0.313   0.7541  
    ## min_positive_polarity         -1.425e+04  1.732e+04  -0.822   0.4111  
    ## max_positive_polarity         -2.682e+03  6.975e+03  -0.384   0.7008  
    ## avg_negative_polarity          1.362e+04  1.913e+04   0.712   0.4767  
    ## min_negative_polarity         -1.324e+03  6.893e+03  -0.192   0.8477  
    ## max_negative_polarity         -2.206e+04  1.624e+04  -1.358   0.1748  
    ## title_subjectivity             1.753e+03  4.101e+03   0.427   0.6692  
    ## title_sentiment_polarity       1.991e+03  4.490e+03   0.443   0.6576  
    ## abs_title_subjectivity         6.136e+03  5.656e+03   1.085   0.2784  
    ## abs_title_sentiment_polarity  -1.904e+03  6.752e+03  -0.282   0.7781  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 24130 on 689 degrees of freedom
    ## Multiple R-squared:  0.06031,    Adjusted R-squared:  -0.00515 
    ## F-statistic: 0.9213 on 48 and 689 DF,  p-value: 0.6261

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

    ## [1] 8182.209

### Boosted Trees

Below is the RMSE of the optimal boosted regression tree:

``` r
boostedTreePred <- predict(boostedFit, newdata = dplyr::select(newsTest, -shares))
sqrt(mean((boostedTreePred-newsTest$shares)^2))
```

    ## [1] 6821.915

### linear regression

Below is the RMSE of the multiple linear model:

``` r
linearPred<-predict(linearfit, newdata=dplyr::select(newsTest, -shares))
sqrt(mean((linearPred-newsTest$shares)^2))
```

    ## [1] 8756.734
