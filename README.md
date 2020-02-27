# Sentiment analysis of hotel reviews

## Problem description
We perform a sentiment analysis of textual reviews of hotel stays. We want to understand whether the users expressed positive or negative feelings in their comments. To do this, we build a binary classification model that is able to predict the sentiment contained in hotel reviews.

We use a dataset that contains more than twenty thousand hotel reviews, scraped from the [tripadvisor.it](https://www.tripadvisor.it/) Italian website. Each review is labeled either positive or negative. The Polytechnic University of Turin has provided this dataset for the Data Science Lab: Process and methods exam project in the academic year 2019/2020.

## Implementation
We use Python 3.7.5 and [Jupyter Notebook](https://jupyter.org/). The following packages are required to run the code. We use the specified package versions.
- [scikit-learn](https://scikit-learn.org/) 0.22
- [NumPy](https://numpy.org/) 1.17.4
- [pandas](https://pandas.pydata.org/) 0.25.3
- [Natural Language Toolkit](https://www.nltk.org/) 3.4.5
- [emoji](https://github.com/carpedm20/emoji) 0.5.4
- [Matplotlib](https://matplotlib.org/) 3.1.1
- [seaborn](https://seaborn.pydata.org/) 0.9.0
- [WordCloud](https://github.com/amueller/word_cloud) 1.5.0

## Data exploration
The dataset contains 28784 labeled reviews. We show its structure in the following table.

|       |                                              text | class |
|------:|--------------------------------------------------:|-------|
|     0 | Non è l'hotel più lussuoso in cui abbia mai so... | pos   |
|     1 | Siamo stati qui per 1 notte prima della nostra... | pos   |
|     2 | Hotel è ben posizionato per visitare Torino. A... | pos   |
|     3 | All'arrivo la cordialità e disponibilità dello... | pos   |
|     4 | Abbiamo soggiornato per due notti alla fine de... | pos   |
|   ... |                                               ... | ...   |
| 28749 | L'hotel è vecchio ma caratteristico e devo dir... | neg   |
| 28750 | Per essere un 4 stelle L la camera era un pò s... | pos   |
| 28751 | Io e mia mamma (di età compresa tra 23 e 62) s... | pos   |
| 28752 | Ci siamo sentiti accolti e coccolati fin dall'... | pos   |
| 28753 | Soggiorno fantastico in una posizione fantasti... | pos   |

We can see that each row of the dataset consists of two fields: `text` contains the review written by the user, and `class` contains a label that specifies the sentiment of the review. We note that the `class` field only contains one of two values, `pos` or `neg`, labelling the sentiment of the review as positive or negative.

A first analysis of the data shows that there are no missing values or empty strings. We plot the distribution of the two classes and we show that the dataset is not well balanced, as most reviews are labeled as positive.

![Class distribution](visual/class_distribution.png)

The dataset has been scraped from the TripAdvisor Italian website, therefore we expect to find comments written in Italian. Further exploration reveals reviews written in other languages such as English and Spanish. Chinese characters are present as well. This suggests that they were originally written in another language and subsequently translated to Italian.

We show the distribution of review length in the following plot. For better visualization, we only take into account reviews with length that falls within the 95th percentile. This means that reviews with more than 1800 characters are not considered.

![Review length distribution](visual/review_length_distribution.png)

We can see that there is a slight difference between the length of positive and negative reviews. It appears that positive reviews are usually more concise than negative reviews, which suggests that users who had a bad experience tend to write more in-depth comments. Furthermore, both distributions are positively skewed, meaning that most users do not write very long reviews. In fact, we find that the majority of reviews have at most about 500 characters.

A number of reviews contain spelling errors. We do not know the amount of misclassification errors caused by wrong spelling, but we assume that most misspelled words do not repeatedly appear in the dataset and therefore will be pruned in the data preprocessing step. We can also find words with repeated characters (e.g., "molto" may be written like "mooolto"). We will handle these words in preprocessing with a basic spell correction algorithm.

Many reviews contain Unicode emojis such as `❤️` and `😠`, and text emoticons such as `:)` and `:(`. These characters express sentiment and we shall consider them when building the classification model.
