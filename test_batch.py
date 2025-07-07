import sys
import torch
import utils 
import numpy as np 

from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


model = utils.NeuralNetwork()
model.load_state_dict(torch.load('./spam_detection.pt'))
model.eval()
 
sentenses = utils.get_data_from_txt(sys.argv[1]) 

sent_transformer = SentenceTransformer('all-mpnet-base-v2')

output_result = []
for sent in tqdm(sentenses): 
    vec = sent_transformer.encode(sent) 
    vec = np.array(vec)
    vec = torch.from_numpy(vec.astype(np.float32))

    input_data = vec.unsqueeze(0)

    output = model(input_data)
    _, predicted = torch.max(output.data, 1)

    # get result
    result = 'not spam' if predicted.cpu() == 0 else 'spam'
    output_result.append(result + ', ' + sent)

dt = datetime.now().strftime(r'%Y%m%d_%H%M%S')
with open('./output_' + dt + '.txt', 'w+') as file:
    for sent in output_result:
        file.write(sent + '\n')

print('output saved to: {}'.format('./output_' + dt + '.txt'))
 