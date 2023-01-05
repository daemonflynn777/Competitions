import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import NearestNeighbors

queries = pd.read_csv('data/queries.csv')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# # base model is pretrained EfficientNetB0
# base_model = EfficientNetB0(weights='imagenet')
# base_model.save('models/EfficientNetB0.h5')

base_model = load_model('weights/EfficientNetB0.h5')
# Customize the model to return features from fully-connected layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


def extract_features(path, model=model):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    feature = model.predict(x, verbose=0)[0]

    return feature / np.linalg.norm(feature)


test_embeddings = []

for idx in tqdm(test.idx):
    test_embeddings.append(extract_features(f'data/test/{idx}.png'))

test_embeddings = np.array(test_embeddings)

queries_embeddings = []

for idx in tqdm(queries.idx):
    queries_embeddings.append(extract_features(f'data/queries/{idx}.png'))

queries_embeddings = np.array(queries_embeddings)

neigh = NearestNeighbors(n_neighbors=10, metric='cosine')
neigh.fit(test_embeddings)

distances, idxs = neigh.kneighbors(queries_embeddings, 10, return_distance=True)

pred_data = pd.DataFrame()
pred_data['score'] = distances.flatten()
pred_data['database_idx'] = [test.idx.iloc[x] for x in idxs.flatten()]
pred_data.loc[:, 'query_idx'] = np.repeat(queries.idx, 10).values

pred_data.to_csv('data/submission.csv', index=False)
