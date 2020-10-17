## Online News Popularity Project

The primary objective of this project is to predict the number of social media shares using two different tree-based algorithms. The first algorithm will be a non-ensemble regression tree, and the second algorithm will be a boosted trees algorithm, which is a state-of-the-art classification technique.

The secondary objective of this project is to create an automated report that outputs the classification analysis for each articles published on each weekday. 

The dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity).

There are 61 attributes, 58 which are predictive attributes, 2 of which are non-predictive (url and timestamp), and 1 that is the target.

The types of veriables include the number of words in the article, the number of words in the title, the positivity and sentimentality of the article, the article's subject, the number of keywords used, and much more.

The required packages to run this analysis are tidyverse, caret, tree, and patchwork.

### Links to Reports
- [Monday report](monday.md)
- [Tuesday report](tuesday.md)
- [Wednesday report](wednesday.md)
- [Thursday report](thursday.md)
- [Friday report](friday.md)
- [Saturday report](saturday.md)
- [Sunday report](sunday.md)

### Instruction to Re-Run Reports
To run the automated production of weekday reports, fork the project and run `Rscript run_report.R`.

`


