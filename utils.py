from torch import nn
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from tqdm import tqdm

import torch
import re
import numpy as np
import string
import nltk

# download words or character list
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)

# initialize class for Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # create MLP Architecture
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512), 
            nn.ReLU(),   
            nn.Linear(512, 256), 
            nn.ReLU(),   
            nn.Linear(256, 128), 
            nn.ReLU(),    
            nn.Linear(128, 64), 
            nn.ReLU(),    
            nn.Linear(64, 32), 
            nn.ReLU(),   
            nn.Linear(32, 16), 
            nn.ReLU(),   
            nn.Linear(16, 8), 
            nn.ReLU(),   
            nn.Linear(8, 4), 
            nn.ReLU(),   
            nn.Linear(4, 2)
        )

    def forward(self, x):
        """
        Feed forward function
        """

        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x 
 

def get_data_from_txt(filepath):
    """
    Import data from text file

    :param: filepath
    :type: string

    :return: array of sentences
    """
    sentence = []
    with open(filepath, 'r') as file:
        sentence.append(file.read().splitlines())
    return sentence[0]


def train_model(trainLoader, epochs=5):
    """
    Train MLP Model

    :param: trainLoader
    :type: torch.utils.data.DataLoader

    :param: epochs
    :type: int

    :return: trained model
    """
    
    # initialize loss function
    loss_function = nn.CrossEntropyLoss()

    # initialize MLP model
    model = NeuralNetwork()

    # initialize optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.0015)

    # train model
    epochs = 15
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(trainLoader):
            inputs, labels = data

            # set optimizer to zero_grad to remove previous epoch gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            # backward propagation
            loss.backward()

            optimizer.step()
            running_loss += loss

        # display statistics
        print(f'Epochs: {epoch}, Loss: {running_loss / 2000:.5f}')
    
    # save model to specific path
    torch.save(
        model.state_dict(),
        r'C:\Users\cj_khoh\Documents\UnifiedComms\Scripts\Python\spam_detection_v3.pt'
    ) 

    return model


def predict(model, testLoader): 
    """
    Make prediction with testing data and a model

    :param: model
    :type: torch.nn.Module

    :param: testLoader
    :type: torch.utils.data.DataLoader

    :return: accuracy of the model
    """
    pred = []
    CM = 0
    correct, total = 0, 0 
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data 

            outputs = model(inputs)
            __, predicted = torch.max(outputs.data, 1)

            CM += confusion_matrix(labels.cpu(), predicted.cpu(), labels=[0, 1])
            pred += predicted.cpu()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    acc = 100 * correct // total

    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall) 

    return acc, precision, recall, f1_score, np.array(pred)


# text cleaning
def clean_text(data):
    """
    perform text cleaning to text, which remove unnecessary character
    """
    
    def all_text_to_lower(text):
        """
        Turn all words in sentences in lower cace

        :param: text
        :type: string
        :return: string
        """
        return text.lower()

    def remove_number(text):
        """
        Remove all number in sentence

        :param: text
        :type: string
        :return: text with no number
        """
        pattern = r'\d+'
        return re.sub(pattern=pattern, repl=' ', string=text)

    def remove_punctuation(text):
        """
        Remove all punctuation in sentence

        :param: text
        :type: string
        :return: text with no punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(text):
        """
        Remove all stopword in sentence

        :param: text
        :type: string
        :return: text with no stopword
        """
        removed = []
        stop_words = list(stopwords.words("english"))
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            if tokens[i] not in stop_words:
                removed.append(tokens[i])
        return " ".join(removed)

    def remove_extra_white_spaces(text):
        """
        Remove all extra whitespace in sentence

        :param: text
        :type: string
        :return: text with no extra white space
        """
        single_char_pattern = r'\s+[a-zA-Z]\s+'
        without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
        return without_sc

    def lemmatizing(text):
        """
        Convert a word to its base form

        :param: text
        :type: string
        :return: base word
        
        """
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            lemma_word = lemmatizer.lemmatize(tokens[i])
            tokens[i] = lemma_word
        return " ".join(tokens)

    # result = data.apply(lambda x : all_text_to_lower(x))
    # result = result.apply(lambda x : remove_number(x))
    # result = result.apply(lambda x : remove_punctuation(x))
    # result = result.apply(lambda x : remove_stopwords(x))
    # result = result.apply(lambda x : remove_extra_white_spaces(x))
    result = data.apply(lambda x : lemmatizing(x))
    
    return result