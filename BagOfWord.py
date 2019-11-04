import nltk


class BagOfWord:

    def createDictionary(self, dataset):
        # Creating the Bag of Words model
        word2count = {}
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word2count.keys():
                    word2count[word] = 1
                else:
                    word2count[word] += 1
