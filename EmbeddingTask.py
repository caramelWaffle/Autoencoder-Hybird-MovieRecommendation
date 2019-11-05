import torch
import numpy
import nltk
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class EmbeddingUtil:
    sentence = ""
    model = "word2vec"
    dictionary = {}

    def __init__(self, sentence, model, dictionary):
        self.sentence = sentence
        self.model = model
        self.dictionary = dictionary

    def get_embedding(self):
        if self.model == 'bert':
            return get_bert_embedding(self.sentence)
        if self.model == 'bow':
            return get_bow_embedding(self.sentence, self.dictionary)


def get_bert_embedding(sentence):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    layer_i = 0
    batch_i = 0
    token_i = 0

    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

    token_embeddings = []

    for token_i in range(len(tokenized_text)):

        hidden_layers = []

        for layer_i in range(len(encoded_layers)):
            vec = encoded_layers[layer_i][batch_i][token_i]
            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)

    sentence_embedding = torch.mean(encoded_layers[11], 1)
    return sentence_embedding.numpy()


def get_bow_embedding(sentence, dictionary):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    item_infomation_matrix = numpy.zeros(len(dictionary))
    stop_words = set(stopwords.words('english'))

    for word in sentence:
        tokens = tokenizer.tokenize(word.lower())
        filtered_token = [w for w in tokens if w not in stop_words]
        for token in filtered_token:
            item_infomation_matrix[list(dictionary.keys()).index(stemmer.stem(lemmatizer.lemmatize(token.lower())))] = 1
    return item_infomation_matrix
