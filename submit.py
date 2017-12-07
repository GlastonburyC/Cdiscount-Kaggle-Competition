from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pandas as pd
import bson
from keras.applications.xception import preprocess_input
import os
from collections import defaultdict
from tqdm import *
import multiprocessing as mp
import struct
import os, sys, math, io
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

num_test_products = 1768182


categories_path = os.path.join("input/category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("categories.csv")
categories_df.head()

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()

submission_df = pd.read_csv("sample_submission.csv")
submission_df.head()

test_bson_path = os.path.join("input/test.bson")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
data = bson.decode_file_iter(open(test_bson_path, "rb"))

model = load_model('Inception.CSdiscount.weights.h5')

with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])
        batch_x = np.zeros((num_imgs, 299, 299, 3), dtype=K.floatx())
        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]
            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(299, 299))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)
            # Add the image to the batch.
            batch_x[i] = x
        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)
        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]        
        pbar.update()

submission_df.to_csv("my_submission.csv.gz", compression="gzip", index=False)
