# Sentiment analysis of hotel reviews

## Problem description
We perform a sentiment analysis of textual reviews of hotel stays. We want to understand whether the users expressed positive or negative feelings in their comments. To do this, we build a binary classification model that is able to predict the sentiment contained in hotel reviews.

We use a dataset that contains more than twenty thousand hotel reviews, scraped from the [tripadvisor.it](tripadvisor.it) Italian website. Each review is labeled either positive or negative. The Polytechnic University of Turin has provided this dataset for the Data Science Lab: Process and methods exam project in the academic year 2019/2020.

## Data exploration
The dataset contains 28784 labeled reviews. Each row of the dataset consists of two fields: `text` contains the review written by the user, and `class` contains a label that specifies the sentiment of the review. We note that the `class` field only contains one of two values, `pos` or `neg`, labelling the sentiment of the review as positive or negative.

A first analysis of the data shows that there are no missing values or empty strings. We plot the distribution of the two classes and we show that the dataset is not well balanced, as most of the reviews are labeled as positive.

The dataset has been scraped from the TripAdvisor Italian website, therefore we expect to find comments written in Italian. Further exploration reveals entire reviews, or parts of them, written in other languages such as English and Spanish. Chinese characters are present as well. This suggests that there are reviews which were originally written in another language and subsequently translated to Italian.

A number of reviews contain spelling errors and words with repeated characters.

The text of reviews contain emojis that express sentiment, which we shall consider in the analysis.

Furthermore, special Unicode characters are present and should be removed.
