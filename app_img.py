import streamlit as st
from PIL import Image
import numpy as np
import pickle
from pickle import dump, load
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
#from gtts import gTTS
from IPython.display import Audio
import csv
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from pylab import imread,subplot,imshow,show
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re 
import cv2
from tensorflow.keras.models import load_model, model_from_json, Sequential
import pandas

@st.cache()
def load_image(imag):
    img = Image.open(imag)
    return img

@st.cache()
def load_values():
    ixtoword = load(open("./ixtoword", "rb"))
    wordtoix = load(open("./wordtoix", "rb"))
    max_length = 34
    vocab_size = 1651
    return (ixtoword,wordtoix,max_length,vocab_size)

#@st.cache()
def model_load():
    modelc = load_model("./modellrtri4.h5")
    modeli = InceptionV3(weights='imagenet')
    modeli = Model(modeli.input,modeli.layers[-2].output)
    return (modelc,modeli)

@st.cache()
def load_fact():
    reader = csv.reader(open('./facts.csv',encoding="utf8"))          
    mydict = {}
    for row in reader:
        k,v = row
        mydict[k]=v
    return (mydict)

def greedySearch(photo,max_length,wordtoix,ixtoword,modelc):
    in_text = 'start'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = modelc.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'end':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final    
    
def facts(cap,mydict):        
        try:
            objs = mydict.keys()
            capwd = (re.sub("[^\w]", " ",cap).split())
            s1 = set(objs)
            s2 = set(capwd)
            cm = s1&s2  
            fact = mydict[list(cm)[0]]
            return fact
        except:
            return "Facts not available"

def call(modeli,max_length,wordtoix,ixtoword,modelc,imag):
        im = imread(imag)
        
        cvt_image = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(cvt_image)
        im_resized = im_pil.resize((299,299))
        im_array = image.img_to_array(im_resized)
        image_array_expanded = np.expand_dims(im_array, axis = 0)
        x = preprocess_input(image_array_expanded)

        #Predicting the caption
        pr = modeli.predict(x)
        cap = greedySearch(pr,max_length,wordtoix,ixtoword,modelc)    
        
        return (cap)      


st.set_page_config(layout="wide")
ixtoword,wordtoix,max_length,vocab_size = load_values()
modelc,modeli = model_load()
mydict = load_fact()
st.write("""
## Image Caption and Facts Generator
""")
try:
    imag = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])
    st.image(imag,width = 300)

    if imag is not None:        
        cap = call(modeli,max_length,wordtoix,ixtoword,modelc,imag)        
        fact = facts(cap,mydict)
        st.write("**Caption:**",cap)
        st.write("**Facts:**",fact)
    
except :
    st.write("Upload a Image")