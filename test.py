from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity

workers = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        print("---- " , dataset.idx_to_class)
        names.append(dataset.idx_to_class[y])


aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
scores = [[cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1)) for e2 in embeddings] for e1 in embeddings]
# print("embed: ", embeddings)
print(pd.DataFrame(scores, columns=names, index=names))


for i in range(len(embeddings)):
    for j in range(len(embeddings)):
        if names[i] == names[j]:
            continue
        e1 = embeddings[i]
        e2 = embeddings[j]
        score = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))
        print("%s vs. %s , similarity = %f " %(names[i], names[j], score))

# print("len of embds: ", len(embeddings))
# print("len of names : ", len(names))
        
