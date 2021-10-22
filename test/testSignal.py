# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:49:01 2019

@author: Bagger
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
import requests
import json



Q = 20
N = 40
M = 20
K = 3

numberPods = 3

# generate the data

# Y = DX
# |x|_0 = n_nonzero_coefs

Y, D, X = make_sparse_coded_signal(n_samples=Q,
                                   n_components=N,
                                   n_features=M,
                                   n_nonzero_coefs=K,
                                   random_state=0)

data = []
split_size = int(np.floor(Q/numberPods))
for i in range(0,numberPods):
    data.append(Y[:, range(split_size*i, split_size*(i+1))])
    data[i] -= np.mean(data[i], axis=0) # normalization
    data[i] /= np.std(data[i], axis=0) # standardizing


D = np.matrix(np.random.rand(M,N))

# %% Send data

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#Make HTTP post
url = 'http://192.168.1.111:30470/load_data/'

payload_D = D.tolist()
payload_Y = json.dumps(data, cls=NumpyEncoder)

payload = {'D': payload_D,'Y': payload_Y}
r = requests.post(url,json=payload)
print(r)

# %% Start work

url = 'http://192.168.1.111:30470/start_work/'
r = requests.post(url)
print(r)


# %% Get training data Respons

response = requests.get('http://192.168.1.111:30654/training_data/')

print(response.content)

# %% Get Results

response = requests.get('http://192.168.1.111:31664/get_results/')

print(response)

D_new = []
X_new = []
stats = []

content = response.json()

for res in response.json():
    D_new.append(res['D'])
    X_new.append(res['X'])
    stats.append(res['S'])
