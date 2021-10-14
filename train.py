import numpy as np
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import FreqDist

seed = 46

# PERSIAPAN DATA
# sastrawi
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
stopwords += ['yg', 'sm', 'www', 'cek', 'hub', 'nama1', 'rp', 'utk', 'no', 'aja']

# buka data
df = pd.read_csv("data/dataset_sms_spam_v1.csv")
df['length'] = df['Teks'].apply(len)

# split data
X_train, X_test, y_train, y_test = train_test_split(df.Teks, 
                                                    df.label, 
                                                    test_size=0.25,
                                                    random_state=seed)


# EKSTRAKSI FITUR
# bag of words
vect = CountVectorizer(stop_words=stopwords)
X_train_vec = vect.fit_transform(X_train)
X_test_vec = vect.transform(X_test)

# MODELLING
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# EVALUASI
train_score = nb.score(X_train_vec, y_train) * 100
test_score = nb.score(X_test_vec, y_test) * 100

# tulis score ke file
with open("metrics.txt", "w") as outfile:
    outfile.write("Skor Training : %2.1f%%\n" %train_score)
    outfile.write("Skor Test : %2.1f%%\n" %test_score)
    

# VISUALISASI

# histogram
df.length.plot(kind='hist', title='Histogram Jumlah Kata')
plt.savefig("histogram_kata.png", dpi=120)
plt.close()

# frekuensi kata
sms_df = df['Teks']

# creating a bag-of-words model
all_words = []
for sms in sms_df:
    words = sms.split()
    for w in words:
        all_words.append(w)
     
all_words = FreqDist(all_words)
all_words.plot(15, title='Top 15 Kata Terbanyak dalam Corpus');
plt.savefig("top15.png", dpi=120)
plt.close()

# wordcloud
sms = ' '.join(sms for sms in df.Teks)
wordcloud = WordCloud(stopwords=stopwords,background_color="white").generate(sms)
plt.imshow(wordcloud)
plt.title("wordcloud")
plt.axis("off")
plt.savefig("wordcloud.png", dpi=120)
plt.close()

