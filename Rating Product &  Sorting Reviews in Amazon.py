
#############################################
# Rating Product & Sorting Reviews in Amazon
#############################################


##################
# Business Problem
##################

"""
One of the most important problems in e-commerce is the correct calculation of
the points given to the products after sales.
Another problem is the correct ordering of the comments given to the products.
With the solution of these problems, the e-commerce site and the sellers will increase
their sales, while the customers will complete the purchasing journey without any problems.
"""


#############################
# Dataset Story and Variables
#############################
"""
This dataset, which includes Amazon product data, includes product categories and various metadata.
The product with the most reviews in category X has user ratings and reviews.
12 Variables    4915 Observations     71.9 MB
# reviewerID            :User ID
# asin                  :Product ID
# reviewerName          :User name
# helpful               :Useful review rating
# reviewText            :Review text
# overall               :Product rating
# summary               :Review summary
# unixReviewTime        :Review time
# reviewTime            :Review time Raw
# day_diff              :Number of days since review
# helpful_yes           :The number of times the review was found useful
# total_vote            :Number of votes given to the review
"""


###############################
# Preparing and Analyzing Data
###############################

import numpy as np
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Projects/Rating Product &  Sorting Reviews in Amazon/amazon_review.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

##################
# Rating Product
##################

df["overall"].mean()
#  4.587589013224822

df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4915 entries, 0 to 4914
# Data columns (total 12 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   reviewerID      4915 non-null   object
#  1   asin            4915 non-null   object
#  2   reviewerName    4914 non-null   object
#  3   helpful         4915 non-null   object
#  4   reviewText      4914 non-null   object
#  5   overall         4915 non-null   float64
#  6   summary         4915 non-null   object
#  7   unixReviewTime  4915 non-null   int64
#  8   reviewTime      4915 non-null   object
#  9   day_diff        4915 non-null   int64
#  10  helpful_yes     4915 non-null   int64
#  11  total_vote      4915 non-null   int64
# dtypes: float64(1), int64(4), object(7)
# memory usage: 460.9+ KB

# Converting reviewTime to datetime
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

#  accept max value of reviewTime as current_date
current_date = df["reviewTime"].max()
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4915 entries, 0 to 4914
# Data columns (total 12 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   reviewerID      4915 non-null   object
#  1   asin            4915 non-null   object
#  2   reviewerName    4914 non-null   object
#  3   helpful         4915 non-null   object
#  4   reviewText      4914 non-null   object
#  5   overall         4915 non-null   float64
#  6   summary         4915 non-null   object
#  7   unixReviewTime  4915 non-null   int64
#  8   reviewTime      4915 non-null   datetime64[ns]
#  9   day_diff        4915 non-null   int64
#  10  helpful_yes     4915 non-null   int64
#  11  total_vote      4915 non-null   int64
# dtypes: datetime64[ns](1), float64(1), int64(4), object(6)
# memory usage: 460.9+ KB

# # Expression of reviews in days
df["days"] = (current_date - df["reviewTime"]).dt.days

q1 = df["days"].quantile(q=0.25)   # 280.0
q2 = df["days"].quantile(q=0.5)    # 430.0
q3 = df["days"].quantile(q=0.75)   # 600.0


# The process of comparing the average of each time period in weighted scoring
df.loc[df["days"] <= q1, "overall"].mean()            # 4.6957928802588995

df.loc[(df["days"] > q1) & (df["days"] <= q2), "overall"].mean()     # 4.636140637775961

df.loc[(df["days"] > q2) & (df["days"] <= q3), "overall"].mean()    # 4.571661237785016

df.loc[(df["days"] > q3), "overall"].mean()

# time based weighted average rating
def time_based_weighted_average(dataframe, w1=40, w2=30, w3=20, w4=10):
    return dataframe.loc[df["days"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > q3), "overall"].mean() * w4 / 100


time_based_weighted_average(df)   # 4.628116998159475

##################
# Sorting Reviews
##################

# Define the helpful_no variable
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head(10)

# score_pos_neg_diff
def score_up_down_diff(yes, no):
    return yes - no

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# conclusion
"""      overall    days    helpful_yes    helpful_no    score_pos_neg_diff  score_average_rating  wilson_lower_bound              
2031     5.00000     701           1952            68                  1884               0.96634             0.95754
3449     5.00000     802           1428            77                  1351               0.94884             0.93652
4212     1.00000     578           1568           126                  1442               0.92562             0.91214
317      1.00000    1032            422            73                   349               0.85253             0.81858
4672     5.00000     157             45             4                    41               0.91837             0.80811
1835     5.00000     282             60             8                    52               0.88235             0.78465
3981     5.00000     776            112            27                    85               0.80576             0.73214
3807     3.00000     648             22             3                    19               0.88000             0.70044
4306     5.00000     822             51            14                    37               0.78462             0.67033
4596     1.00000     806             82            27                    55               0.75229             0.66359
315      5.00000     846             38            10                    28               0.79167             0.65741
1465     4.00000     237              7             0                     7               1.00000             0.64567
1609     5.00000     256              7             0                     7               1.00000             0.64567
4302     5.00000     261             14             2                    12               0.87500             0.63977
4072     5.00000     758              6             0                     6               1.00000             0.60967
1072     5.00000     941              5             0                     5               1.00000             0.56552
2583     5.00000     488              5             0                     5               1.00000             0.56552
121      5.00000     942              5             0                     5               1.00000             0.56552
1142     5.00000     306              5             0                     5               1.00000             0.56552
1753     5.00000     776              5             0                     5               1.00000             0.56552
"""