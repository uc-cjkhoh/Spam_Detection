import torch
import utils
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = utils.NeuralNetwork()
model.load_state_dict(torch.load('./spam_detection_v2.pt'))
model.eval()

# model = pickle.load(open('D:\Others\Model\spam_detection_svm2.sav', 'rb'))

while True:
    sent = input('Enter a sentence: ')

    if sent == 'exit()':
        break

    sent_transformer = SentenceTransformer('all-mpnet-base-v2')
    sent = sent_transformer.encode(sent)
    
    # sent = sent.reshape(1, -1) 
    # print(model.predict(sent))
    
    sent = np.array(sent)
    sent = torch.from_numpy(sent.astype(np.float32))

    input_data = sent.unsqueeze(0)

    output = model(input_data)
    _, predicted = torch.max(output.data, 1)

    if predicted.cpu() == 0:
        print('This is not a spam')
    else:
       print('This is a spam')
