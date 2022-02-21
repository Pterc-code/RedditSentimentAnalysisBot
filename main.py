from scipy.stats import pearsonr
import praw

import config
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def login():
    r = praw.Reddit(
        username=config.username,
        password=config.password,
        client_id=config.client_id,
        client_secret=config.client_secret,
        user_agent='???'
    )
    return r


def calcPolarSubject(r, subreddit: str) -> list[
    dict[str:tuple[(float, float)]]]:
    ldict = []
    for submission in r.subreddit(subreddit).hot(limit=2):
        rdict = {}
        submission.comments.replace_more(limit=0)
        comments = submission.comments.list()
        for comment in comments:
            analysis = TextBlob(comment.body)
            rdict[comment] = (analysis.subjectivity,
                              analysis.polarity)  # Polarity: how positive the comment is, Subjectivity: How much it is an opinion
        ldict.append(rdict.copy())
        rdict.clear()
    return ldict


def calcRegression(data: dict[str:tuple[(float, float)]]):
    polarity = []
    subjectivity = []
    for i in range(len(data)):
        for comment in data[i]:
            polarity.append(data[i][comment][1])
            subjectivity.append(data[i][comment][0])

    X = np.array(polarity).reshape((-1, 1))
    Y = np.array(subjectivity)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()


def calcBoxplot(data: dict[str:tuple[(float, float)]]):
    polarity = []
    for i in range(len(data)):
        for comment in data[i]:
            polarity.append(data[i][comment][1])
    plt.boxplot(polarity)
    plt.show()


def calcCorrelation(data):
    polarity = []
    subjectivity = []
    for i in range(len(data)):
        for comment in data[i]:
            polarity.append(data[i][comment][1])
            subjectivity.append(data[i][comment][0])
    corr, _ = pearsonr(polarity, subjectivity)
    print('Pearsons correlation: %.3f' % corr)


if __name__ == '__main__':
    subreddit = input('Which subreddit do you want to analyze: ')
    r = login()
    data = calcPolarSubject(r, subreddit)
    calcRegression(data)
    calcBoxplot(data)
    calcCorrelation(data)






































