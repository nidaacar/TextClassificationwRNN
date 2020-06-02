from nltk.corpus import stopwords
from data import train_data, test_data
from model import Model
import nltk
nltk.download('stopwords')
# region PreProcessing

list_of_stopwords = list(stopwords.words("English"))
list_of_stopwords.extend(['.', ',', ':'])
list_of_stopwords.remove('not')

# print(len(list_of_stopwords))


def delete_stop_words(data):
    return_data = dict()
    for key, value in data.items():
        words_tokens = list(key.split(" "))
        st = ''
        w = [word for word in words_tokens if word.lower() not in list_of_stopwords]
        st = " ".join(w)
        return_data[st] = value

    return return_data


test_data = delete_stop_words(test_data)
train_data = delete_stop_words(train_data)

# endregion

rnn_model = Model(train_data, test_data)
info = rnn_model.train(plot_results=True)

# Test the solution

st = ["Ouv shit this is disgusting",
      "i am happy", "not well", "well enough", "such a strange things",
      "this is boring", "this is such a nice"]

for index, value in enumerate(st):
    print(f"Test {index}: {value}")
    r = rnn_model.predict(value)
    print()
