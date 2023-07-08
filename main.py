from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from utils import preprocess, cluster_to_img, iter
from glob import glob
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import config

def create_embeddings(embeddings, i):
  embeddings[i] = model.predict(preprocess(image_paths[i]))
  return embeddings

if __name__ == "__main__":
  path = config.path
  n_samples = config.n_samples
  n_clusters = config.n_clusters
  p = config.p
  alpha = config.alpha

  image_paths = glob(path)    
  # VGG16 is used as pretrained CNN model
  vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, 3))
  model = keras.Sequential()
  
  for layer in vgg_model.layers:
    model.add(layer)
  
  model.add(Flatten())
  model.add(Dense(512, name='embeddings'))
  
  Parallel(n_jobs = -1, prefer='threads')(delayed(create_embeddings)(embeddings, i) for i in tqdm(range(len(image_paths)), position=0, leave=True))
  
  # embeddings = np.loadtxt("gdrive/MyDrive/embeds.txt")
  embeddings = np.squeeze(np.array(embeddings))
  np.savetxt("embeds.txt", embeddings)
  
  km = KMeans(n_clusters=n_clusters).fit(embeddings)
  clusters = np.zeros(n_samples)
  
  for cluster in range(n_clusters):
    for image in np.where(km.labels_ == cluster):
      clusters[image] = cluster

  summ=0
  for i in range(n):
    summ += p*(alpha**i)
  
  prev_votes = np.floor(n_clusters*np.random.rand(4, n)-0.000001)
  boundary = np.zeros(n)
  
  uvote = np.random.randint(n_clusters,size=4)
  
  while(1):
      plt.subplot(2,2,1)
      plt.imshow(cluster_to_img(uvote[0]))
      plt.subplot(2,2,2)
      plt.imshow(cluster_to_img(uvote[1]))
      plt.subplot(2,2,3)
      plt.imshow(cluster_to_img(uvote[2]))
      plt.subplot(2,2,4)
      plt.imshow(cluster_to_img(uvote[3]))
      plt.show()
      
      time.sleep(0.5)
      ip = list(map(int, input("Enter your votes(1 for positive, -1 for negative and 0 to exit): ").split()))
  
      if ip[0] == 0:
        break
        
      uvote = np.array(ip) * uvote
      neg = (uvote < 0).sum()
  
      if neg == 4:
        uvote = np.random.randint(n_clusters,size=4)
      elif neg == 3:
        temp = uvote[uvote > 0][0]
        uvote = np.array([temp, temp, temp, temp])
      elif neg == 2:
        temp1 = uvote[uvote > 0][0]
        temp2 = uvote[uvote > 0][1]
        uvote = np.array([temp1, temp2, temp1, temp2])
      elif neg == 1:
        temp1 = uvote[uvote > 0][0]
        temp2 = uvote[uvote > 0][1]
        temp3 = uvote[uvote > 0][2]
        uvote = np.array([temp1, temp2, temp3, temp3])
  
      uvote, prev_votes = iter(prev_votes, uvote)
