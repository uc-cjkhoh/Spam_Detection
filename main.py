import numpy as np
import pandas as pd
import utils
import torch  

from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer 

# class for input data strcuture
class Data(Dataset):
    def __init__(self, x_train, y_train):
        self.x = torch.from_numpy(x_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len 


if __name__ == '__main__':
    # get dataset
    data = pd.read_csv('C:/Users/cj_khoh/Documents/UnifiedComms/Data/spam_ham_dataset.csv', encoding='ISO-8859-1')
    data = pd.DataFrame(data)
    data = data[['label', 'text']] 
    # data['text'] = utils.clean_text(data['text'])
  
    # text embedding model
    text_embedding_model = SentenceTransformer('all-mpnet-base-v2')

    # separate train and test data
    Y = np.array(data.label)
    Y = pd.factorize(Y)[0]

    X = data.text
    
    # embed train sentence
    embedded_x = []
    for sent in tqdm(X):
        embedded_x.append(text_embedding_model.encode(sent))    
    embedded_x = np.array(embedded_x) 
 
    # over-sampling
    X, Y = RandomOverSampler().fit_resample(embedded_x, Y)
    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=20, shuffle=True)
 
    # group data
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    # initialize batch data for nn_model
    batch_size = 32
    trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # train model
    model = utils.train_model(trainLoader).eval()

    # initializ batch data for testing
    testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # predict
    acc, precision, recall, f1_score, pred = utils.predict(model, testLoader) 
    
    print(f'Accuracy: {acc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1_score}')
    