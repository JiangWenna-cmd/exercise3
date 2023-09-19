import nltk
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

moby_dick = gutenberg.raw('melville-moby_dick.txt')

tokens = nltk.word_tokenize(moby_dick)

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_tags = [(word, 'NN' if tag == 'JJ' else tag) for (word, tag) in pos_tags]

pos_freq = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, tag) for (word, tag) in pos_tags][:20]

pos_freq.plot(30, cumulative=False)
plt.show()


print("Top 5 POS and their frequencies:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print("\nTop 20 lemmas:")
for lemma in lemmas:
    print(lemma)