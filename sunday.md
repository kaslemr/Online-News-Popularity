Online News Analysis - Predicting Media Shares by Article
Characteristics
================
Matt Kasle
10/15/2020

sunday articles
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

    ## [1] 824

Summary of response variable:

    summary(newsTrain$shares)

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      91    1200    1900    3924    3700   64500

It is important to make note of the minimum and maximum of the response
variable, as well as the median and quartiles. For future analysis, it
may be best to remove outliers.

Distribution of response variable:

    ggplot(data = newsTrain, aes(x = shares)) +
      geom_histogram() + xlab("Shares") + ggtitle(paste("Distribution of Shares in Training Data - ",
                                                        params$day_of_week))

![](sunday_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Below is a plot of four interesting variables to the response variable,
shares. We want to see if there are any intersting patterns here.

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

![](sunday_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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

![](sunday_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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
    ##  [1] "self_reference_min_shares" "kw_min_avg"               
    ##  [3] "avg_positive_polarity"     "kw_max_avg"               
    ##  [5] "rate_negative_words"       "LDA_03"                   
    ##  [7] "LDA_04"                    "avg_negative_polarity"    
    ##  [9] "kw_max_min"                "kw_avg_max"               
    ## Number of terminal nodes:  12 
    ## Residual mean deviance:  28620000 = 2.324e+10 / 812 
    ## Distribution of residuals:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  -22120   -2043    -936       0     264   44880

    pruneFit <- cv.tree(treeFit,
                      K=nrow(newsTrain)-1
                      )


    pruneFitFinal <- prune.tree(treeFit, best = pruneFit$size[1]) 

    plot(pruneFit$size ,pruneFit$dev ,type="b")

![](sunday_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
    ## 1       0.1                 1             10      50 6266.247 0.04460433
    ## 4       0.1                 2             10      50 6274.723 0.05515840
    ## 7       0.1                 3             10      50 6308.617 0.04625092
    ## 2       0.1                 1             10     100 6339.620 0.04363465
    ## 5       0.1                 2             10     100 6409.929 0.04772949
    ## 8       0.1                 3             10     100 6429.420 0.04357450
    ## 3       0.1                 1             10     150 6410.235 0.03856114
    ## 6       0.1                 2             10     150 6473.355 0.04487700
    ## 9       0.1                 3             10     150 6518.688 0.04037359
    ##        MAE   RMSESD RsquaredSD    MAESD
    ## 1 3333.600 1584.036 0.06655842 457.5225
    ## 4 3354.224 1504.153 0.09106202 424.3293
    ## 7 3381.136 1509.425 0.07336252 432.3686
    ## 2 3416.257 1556.192 0.06570047 455.4007
    ## 5 3460.574 1470.997 0.07419314 451.6148
    ## 8 3501.758 1454.633 0.05691881 424.5925
    ## 3 3480.430 1526.450 0.05542399 461.7136
    ## 6 3547.440 1459.391 0.06934638 458.0592
    ## 9 3592.120 1415.080 0.05214736 430.1174

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

    ## [1] 7099.371

### Boosted Trees

Below is the RMSE of the optimal boosted regression tree:

    boostedTreePred <- predict(boostedFit, newdata = dplyr::select(newsTest, -shares))
    sqrt(mean((boostedTreePred-newsTest$shares)^2))

    ## [1] 6314.956
