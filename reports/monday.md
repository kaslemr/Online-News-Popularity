Online News Analysis - Predicting Media Shares by Article
Characteristics
================
Matt Kasle
10/15/2020

monday articles
===============

Introduction
------------

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

Data
----

First, we need to read in the data and set up the subsetting of the data
into a single weekday. Then, we’ll split the data set to a training and
test set for training and evaluating the classificaiton models. A 70/30
train-test split will be used.

    set.seed(1)
    library(tidyverse)
    library(caret)
    library(tree)

    news <- read_csv("OnlineNewsPopularity.csv")

    varDayOfWeek <- parse(text=paste0("weekday_is_", params$day_of_week))
    news <- news %>% filter(eval(varDayOfWeek) == 1)

    news <- news %>% select(-url, -timedelta, -weekday_is_monday, -weekday_is_tuesday, -weekday_is_wednesday,
                            -weekday_is_thursday,-weekday_is_friday, 
                            -weekday_is_saturday, -weekday_is_sunday, -is_weekend)

    newsIndex <- createDataPartition(news$shares, p = 0.3, list = FALSE)
    newsTrain <- news[newsIndex, ]

    ## Warning: The `i` argument of ``[`()` can't be a matrix as of tibble 3.0.0.
    ## Convert to a vector.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_warnings()` to see where this warning was generated.

    newsTest <- news[-newsIndex, ]

Summarizations
--------------

Below are summary statistics of the online news data set, such as the
number of observations in the train set, a numerical summary of the
response variable (number of social media shares), a distribution of the
response, and relationships between the response and interesting
variables in the dataset.

Number of rows in training set:

    nrow(newsTrain)

    ## [1] 2000

Summary of response variable:

    summary(newsTrain$shares)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##       4     919    1400    3742    2700  690400

It is important to make note of the minimum and maximum of the response
variable, as well as the median and quartiles. For future analysis, it
may be best to remove outliers.

Distribution of response variable:

    ggplot(data = newsTrain, aes(x = shares)) +
      geom_histogram() + xlab("Shares") + ggtitle(paste("Distribution of Shares in Training Data - ",
                                                        params$day_of_week))

![](reports/monday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Below is a plot of four interesting variables to the response variable,
shares. We want to see if there are any intersting patterns here.

    library(patchwork)

    ## Warning: package 'patchwork' was built under R version 3.6.2

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

![](reports/monday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

There are six different article subjects. It would be interesting to
know if some subjects are more popular than others, so we can plot the
median shares by the different article types.

    news['article_type'] <- ifelse(news$data_channel_is_lifestyle == 1, "Lifestyle",
           ifelse(news$data_channel_is_entertainment == 1, "Entertainment",
           ifelse(news$data_channel_is_world == 1, "World",
           ifelse(news$data_channel_is_bus == 1, "Bus",
           ifelse(news$data_channel_is_socmed == 1, "SocMed",
           ifelse(news$data_channel_is_tech == 1, "Tech","Other"))))))

    ggplot(data=news, aes(x = article_type, y = shares)) + geom_bar(stat = "summary", fun.y = "median") +
        labs(x="Article Type", y="Median Shares", title="Median Shares by Article Type")

![](reports/monday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

    # drop column used for plotting
    news <- news %>% select(-article_type)

Modeling
--------

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

    treeFit <- tree(shares ~ ., data = newsTrain)
    summary(treeFit)

    ## 
    ## Regression tree:
    ## tree(formula = shares ~ ., data = newsTrain)
    ## Variables actually used in tree construction:
    ## [1] "kw_avg_min"            "kw_min_avg"            "min_negative_polarity"
    ## Number of terminal nodes:  4 
    ## Residual mean deviance:  245100000 = 4.891e+11 / 1996 
    ## Distribution of residuals:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -160500.0   -2326.0   -1839.0       0.0    -539.3  528800.0

    pruneFit <- cv.tree(treeFit,
                      K=nrow(newsTrain)-1
                      )


    pruneFitFinal <- prune.tree(treeFit, best = pruneFit$size[1]) 

    plot(pruneFit$size ,pruneFit$dev ,type="b")

![](reports/monday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

### Boosted tree model

Next, we’ll fit a boosted regression tree. The boosted tree algorithm
has a few hyperparameters, and we’ll use repeated 10-fold
cross-validation to determine the optimal hyperparameter values. The
hyperparameters of the optimal boosted tree is printed below, as well a
summary of each fitted model.

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

    ##   n.trees interaction.depth shrinkage n.minobsinnode
    ## 1      50                 1       0.1             10

    boostedFit$results

    ##   shrinkage interaction.depth n.minobsinnode n.trees     RMSE   Rsquared
    ## 1       0.1                 1             10      50 12934.69 0.01657501
    ## 4       0.1                 2             10      50 13118.17 0.01809484
    ## 7       0.1                 3             10      50 13050.81 0.02427935
    ## 2       0.1                 1             10     100 13095.39 0.01429058
    ## 5       0.1                 2             10     100 13281.27 0.01764274
    ## 8       0.1                 3             10     100 13262.21 0.01696909
    ## 3       0.1                 1             10     150 13096.10 0.01339196
    ## 6       0.1                 2             10     150 13510.93 0.01491678
    ## 9       0.1                 3             10     150 13461.57 0.01662168
    ##        MAE   RMSESD RsquaredSD    MAESD
    ## 1 3928.494 12483.75 0.03507784 1041.924
    ## 4 3980.116 12328.95 0.02680138 1029.512
    ## 7 4011.278 12367.78 0.03613895 1036.511
    ## 2 3990.731 12404.41 0.03006098 1041.272
    ## 5 4085.441 12299.41 0.02333652 1049.882
    ## 8 4149.086 12351.84 0.02635892 1044.624
    ## 3 3984.889 12404.97 0.02800575 1046.104
    ## 6 4180.347 12225.15 0.02096103 1040.676
    ## 9 4279.897 12306.35 0.02576599 1054.394

Model Evaluations
-----------------

Finally, we’ll evaluate the performance of each model by seeking the
lowest root mean squared error of its predictions on the test dataset
when compared to the actual values in the dataset. This should be a good
approximation of the model’s performance on unseen data.

### Regression Tree

Below is the RMSE of the optimal (non-ensemble) regression tree:

    treePred <- predict(pruneFitFinal, newdata = dplyr::select(newsTest, -shares))
    sqrt(mean((treePred-newsTest$shares)^2))

    ## [1] 15593.79

### Boosted Trees

Below is the RMSE of the optimal boosted regression tree:

    boostedTreePred <- predict(boostedFit, newdata = dplyr::select(newsTest, -shares))
    sqrt(mean((boostedTreePred-newsTest$shares)^2))

    ## [1] 13551.85
