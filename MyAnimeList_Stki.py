#!/usr/bin/env python
# coding: utf-8

# In[2]:


## UAS SISTEM TEMU KEMBALI INFORMASI (STKI)

## ANGGOTA 1  : Didit Iswantoro (17.01.53.0030)
## ANGGOTA 2  : Ahmad Samsul Muarif (17.01.53.0037)
## ANGGOTA 3  : Ardi Kurniawan (17.01.53.0071)
## Kelas    : B1


# In[3]:


pip install bs4


# In[4]:


pip install wordcloud


# In[5]:


pip install seaborn


# In[6]:


import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')
# Mengimport Library Beautifulsoup
from bs4 import BeautifulSoup
import requests


# In[30]:


# Mengambil data sebanyak 100
start = 2033
end = 2133
array = []

for start in range(int(start), int(end)+1):
    # Merequest sumber halaman web
    page = requests.get("https://myanimelist.net/anime/" + str(start))

    # Merapikan tampilan hasil dari web yang di scraping 
    scraping = BeautifulSoup(page.content,'html.parser')

    # Sraping Title/Judul Anime
    nyoba1 = scraping.find('title')
    
    if nyoba1 is None:
        nyoba1 = "-"
    else:
        nyoba1 = nyoba1.text.replace(" - MyAnimeList.net", "").strip()

    # Scraping Popularity Anime
    nyoba4 = scraping.find('span',attrs={"class":"numbers popularity"})

    if nyoba4 is None:
        nyoba4 = 0
    else:
        nyoba4 = nyoba4.text.replace("Popularity #", "").strip()

    # Scraping Rating Anime
    nyoba5 = scraping.find('span',attrs={"itemprop":"ratingValue"})

    if nyoba5 is None:
        nyoba5 = 0
    else:
        nyoba5 = nyoba5.text

    # Scraping Deskripsi Anime
    nyoba6 = scraping.find('p',attrs={"itemprop":"description"})

    if nyoba6 is None:
        nyoba6 = "-"
    else:
        nyoba6 = nyoba6.text.strip()

    array.append([int(start), str(nyoba1), float(nyoba5), int(nyoba4), str(nyoba6)])
   
print(array)


# In[31]:


# mengambil data frame
anime_list = pd.DataFrame(array, columns=['anime_id', 'anime_title', 'anime_rating', 'anime_popularity', 'anime_description'])
# Menampilkan data frame yang sudah di ambil
print(anime_list)


# In[44]:


# menampilkan data frame dalam bentuk tabel
data = pd.DataFrame(anime_list)
data


# In[45]:


# Menghapus data yang tidak digunakan yaitu anime id dan anime desckripsi
data = data.drop(['anime_id', 'anime_description'], axis = 1)
data.head()


# In[47]:


# Mengambil data yang akan di klastering yaitu anime rating dan anime popularity
data_x = data.iloc[:, 1:3]
data_x.head()


# In[50]:


# Melihat persebaran data menggunakan fungsi seaborn
sns.scatterplot(x="anime_rating", y="anime_popularity", data=data, s=50, color="red", alpha = 0.5)


# In[51]:


# Mengubah variabel berbentuk data frame menjadi aray
x_array = np.array(data_x)
print(x_array)


# In[55]:


# Melakukan proses standarisasi pada data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled


# In[57]:


# Menentukan jumlah klaster sebanyak 5
kmeans = KMeans(n_clusters = 5, random_state=123)
# Menentukan klaster dari data
kmeans.fit(x_scaled)


# In[59]:


# Mencari nilai pusat dari klaster
print(kmeans.cluster_centers_)


# In[61]:


# Menampilkan hasil klaster
print(kmeans.labels_)


# In[63]:


# Menampilkan data klaster
data["cluster"] = kmeans.labels_
data.head()


# In[65]:


# Memvisualisasi data klaster menggunakan library matplotlib
fig, ax = plt.subplots()
sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 50,
c = data.cluster, marker = "o", alpha = 0.5)
centers = kmeans.cluster_centers_
ax.scatter(centers[:,1], centers[:,0], c='blue', s=100, alpha=0.5);plt.title("Hasil Clustering Menggunakan K-Means")
plt.xlabel("Scaled Anime Rating")
plt.ylabel("Scaled Anime Popularity")
plt.show()


# In[ ]:




