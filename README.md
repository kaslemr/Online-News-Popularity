## Online News Popularity Project

The primary objective of this project is to predict the number of social media shares using two different tree-based algorithms. The first algorithm will be a non-ensemble regression tree, and the second algorithm will be a boosted trees algorithm, which is a state-of-the-art classification technique.

The secondary objective of this project is to create an automated report that outputs the classification analysis for each articles published on each weekday. 

The dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal is to predict the number of shares in social networks (popularity).

There are 61 attributes, 58 which are predictive attributes, 2 of which are non-predictive (url and timestamp), and 1 that is the target.

The types of veriables include the number of words in the article, the number of words in the title, the positivity and sentimentality of the article, the article's subject, the number of keywords used, and much more.

The required packages to run this analysis are tidyverse, caret, tree, and patchwork.

In the README.md file for the repo, give a brief description of the purpose of the repo and create links to each sub-document (Monday’s analysis, Tuesday’s analysis, etc.). Links can be made to the sub-documents using relative paths. For instance, if you have all of the outputted .md files in the main directory you would just use markdown linking:
• The analysis for [Monday is available here](MondayAnalysis.md).
Of course, this supports the use of folders as well if you output the files into separate folders.
• You should also make a note of all packages required to run your analysis here.
• You should include the code used to automate the process (i.e. the render function you used) here as well.
You can use the [editor on GitHub](https://github.com/kaslemr/Online-News-Popularity/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

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


