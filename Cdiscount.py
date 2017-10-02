# -*- coding: utf-8 -*-

import io
import os
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout
from keras.utils import np_utils
#preprocessing
data = bson.decode_file_iter(open('train_example.bson', 'rb'))

prod_to_category = dict()
picture=dict()

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    prod_to_category[product_id] = category_id
    for e, pic in enumerate(d['imgs']):
        picture[product_id] = imread(io.BytesIO(pic['picture']))
        # do something with the picture, etc

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

# creating a new column in category_name dataframe so that later this column can be one hot encoded
#category_id is our dependent variable
category_names = pd.read_csv('category_names.csv',index_col='category_id')
category_names.head()
category_names['category_idx']= pd.Series(range(len(category_names)),index=category_names.index)

pd.DataFrame(picture,index=prod_to_category.category_id)

defaultdict
#model


model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape(180,180,3),activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))




import os
import sys
import numpy as np # linear algebra
import pandas as pd
import bson
import cv2
import matplotlib.pyplot as plt

