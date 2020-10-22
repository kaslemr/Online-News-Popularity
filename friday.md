Online News Analysis - Predicting Media Shares by Article
Characteristics
================
Matt Kasle
10/15/2020

# friday articles

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

    ## [1] 1711

Summary of response variable:

``` r
summary(newsTrain$shares)
```

    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ##     70.0    973.5   1500.0   3218.1   2700.0 104100.0

It is important to make note of the minimum and maximum of the response
variable, as well as the median and quartiles. For future analysis, it
may be best to remove outliers.

Distribution of response variable:

``` r
ggplot(data = newsTrain, aes(x = shares)) +
  geom_histogram() + xlab("Shares") + ggtitle(paste("Distribution of Shares in Training Data - ",
                                                    params$day_of_week))
```

![](friday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](friday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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

![](friday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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
    ## [1] "kw_avg_avg"                "self_reference_min_shares" "n_non_stop_unique_tokens"  "n_tokens_title"           
    ## [5] "n_unique_tokens"           "kw_avg_max"                "avg_positive_polarity"    
    ## Number of terminal nodes:  9 
    ## Residual mean deviance:  39160000 = 6.665e+10 / 1702 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -38000.00  -1728.00  -1278.00      0.00     22.42  82880.00

``` r
pruneFit <- cv.tree(treeFit,
                  K=nrow(newsTrain)-1
                  )


pruneFitFinal <- prune.tree(treeFit, best = pruneFit$size[1]) 

plot(pruneFit$size ,pruneFit$dev ,type="b")
```

![](friday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
    ## 1       0.1                 1             10      50 6657.000 0.03395868 2862.997 2403.522 0.03074098 452.6250
    ## 4       0.1                 2             10      50 6772.467 0.02690501 2891.136 2391.239 0.02663976 457.0089
    ## 7       0.1                 3             10      50 6822.695 0.02640529 2916.873 2362.163 0.02861856 430.3234
    ## 2       0.1                 1             10     100 6688.743 0.03337363 2884.703 2368.378 0.03163360 446.6308
    ## 5       0.1                 2             10     100 6871.507 0.02444774 2953.717 2340.634 0.02773099 446.7605
    ## 8       0.1                 3             10     100 6960.542 0.02334665 3009.906 2310.783 0.02730471 431.1012
    ## 3       0.1                 1             10     150 6717.032 0.03218936 2904.977 2363.676 0.03122142 448.1834
    ## 6       0.1                 2             10     150 6950.738 0.02350822 3020.392 2308.608 0.02844442 449.8518
    ## 9       0.1                 3             10     150 7067.250 0.02112297 3109.150 2263.746 0.02752012 442.2882

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
    ## -14883  -2173   -954    260  98350 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    2.417e+03  2.629e+03   0.919 0.358002    
    ## n_tokens_title                -3.675e+01  8.299e+01  -0.443 0.657973    
    ## n_tokens_content              -2.704e-01  7.501e-01  -0.361 0.718515    
    ## n_unique_tokens                4.710e+03  5.715e+03   0.824 0.409988    
    ## n_non_stop_words              -3.943e+03  4.853e+03  -0.813 0.416554    
    ## n_non_stop_unique_tokens       5.960e+02  5.001e+03   0.119 0.905156    
    ## num_hrefs                      1.048e+01  1.836e+01   0.571 0.568214    
    ## num_self_hrefs                -2.426e+01  6.375e+01  -0.381 0.703615    
    ## num_imgs                       2.664e+01  2.751e+01   0.968 0.332942    
    ## num_videos                    -2.758e+01  5.190e+01  -0.531 0.595183    
    ## average_token_length          -4.574e+02  7.231e+02  -0.633 0.527097    
    ## num_keywords                  -9.067e+01  1.104e+02  -0.821 0.411677    
    ## data_channel_is_lifestyle     -1.341e+03  1.247e+03  -1.075 0.282380    
    ## data_channel_is_entertainment -1.346e+03  7.704e+02  -1.747 0.080837 .  
    ## data_channel_is_bus           -1.911e+03  1.150e+03  -1.661 0.096811 .  
    ## data_channel_is_socmed        -1.070e+03  1.123e+03  -0.953 0.340908    
    ## data_channel_is_tech          -7.635e+02  1.096e+03  -0.697 0.485949    
    ## data_channel_is_world         -2.626e+03  1.126e+03  -2.331 0.019847 *  
    ## kw_min_min                     2.396e+00  4.904e+00   0.489 0.625251    
    ## kw_max_min                    -2.890e-01  1.990e-01  -1.452 0.146606    
    ## kw_avg_min                     2.518e-01  1.210e+00   0.208 0.835150    
    ## kw_min_max                    -7.296e-04  3.022e-03  -0.241 0.809227    
    ## kw_max_max                    -5.872e-04  1.777e-03  -0.330 0.741160    
    ## kw_avg_max                    -3.425e-03  2.478e-03  -1.382 0.167071    
    ## kw_min_avg                    -3.397e-01  2.307e-01  -1.472 0.141083    
    ## kw_max_avg                    -6.312e-02  8.048e-02  -0.784 0.432980    
    ## kw_avg_avg                     1.762e+00  4.580e-01   3.848 0.000123 ***
    ## self_reference_min_shares      3.831e-03  2.413e-02   0.159 0.873850    
    ## self_reference_max_shares     -1.840e-02  1.029e-02  -1.789 0.073759 .  
    ## self_reference_avg_sharess     2.395e-02  3.068e-02   0.781 0.435130    
    ## LDA_00                         5.817e+02  1.366e+03   0.426 0.670282    
    ## LDA_01                         2.925e+02  1.525e+03   0.192 0.847893    
    ## LDA_02                         1.772e+03  1.366e+03   1.297 0.194857    
    ## LDA_03                        -9.546e+02  1.466e+03  -0.651 0.515022    
    ## LDA_04                                NA         NA      NA       NA    
    ## global_subjectivity            1.975e+03  2.450e+03   0.806 0.420437    
    ## global_sentiment_polarity     -2.990e+03  4.861e+03  -0.615 0.538607    
    ## global_rate_positive_words    -1.798e+04  2.170e+04  -0.828 0.407543    
    ## global_rate_negative_words    -6.828e+03  4.031e+04  -0.169 0.865505    
    ## rate_positive_words            2.043e+03  3.309e+03   0.617 0.537038    
    ## rate_negative_words                   NA         NA      NA       NA    
    ## avg_positive_polarity          1.721e+03  4.169e+03   0.413 0.679692    
    ## min_positive_polarity         -3.669e+03  3.365e+03  -1.090 0.275753    
    ## max_positive_polarity          1.470e+03  1.278e+03   1.150 0.250378    
    ## avg_negative_polarity          4.749e+03  3.656e+03   1.299 0.194132    
    ## min_negative_polarity         -2.660e+03  1.367e+03  -1.946 0.051816 .  
    ## max_negative_polarity         -1.231e+03  3.008e+03  -0.409 0.682275    
    ## title_subjectivity            -6.899e+02  8.891e+02  -0.776 0.437862    
    ## title_sentiment_polarity       3.956e+02  7.448e+02   0.531 0.595364    
    ## abs_title_subjectivity        -1.525e+02  1.064e+03  -0.143 0.886096    
    ## abs_title_sentiment_polarity   8.700e+02  1.254e+03   0.694 0.487807    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6930 on 1662 degrees of freedom
    ## Multiple R-squared:  0.07087,    Adjusted R-squared:  0.04404 
    ## F-statistic: 2.641 on 48 and 1662 DF,  p-value: 1.266e-08

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

    ## [1] 9051.817

### Boosted Trees

Below is the RMSE of the optimal boosted regression tree:

``` r
boostedTreePred <- predict(boostedFit, newdata = dplyr::select(newsTest, -shares))
sqrt(mean((boostedTreePred-newsTest$shares)^2))
```

    ## [1] 8498.582

### linear regression

Below is the RMSE of the multiple linear model:

``` r
linearPred<-predict(linearfit, newdata=dplyr::select(newsTest, -shares))
sqrt(mean((linearPred-newsTest$shares)^2))
```

    ## [1] 8531.011
