from tensorflow import keras
import tensorflow as tf

import json
import transformers
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn
import pandas as pd
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
model=load_model('chatbotdeepfake2.h5')
app=FastAPI()

with open('intents11 (2).json') as intents:
    data=json.load(intents)
    
class train():
    tags=[]
    inputs=[]
    responses={}
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tags.append(intent['tag'])
    df=pd.DataFrame({'Inputs':inputs,'tags':tags})
   
   
    
   
    le=LabelEncoder()

    le.fit_transform(df['tags'])  
    

   


class UserInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True 


 


    
    prediction_input:str

    

    
    


    


   


   # prediction_input=np.int.decode(prediction_input,encoding='cp037')
   
    
    #prediction_input = json.dumps(prediction_input)
   

    

   
 


    #prediction_input=pad_sequences([prediction_input],input_shape)
   
    #user_input=pad_sequences([json_data],)
@app.get('/')
async def index():
    return {"hola"}
import random
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_data(tokenizer, texts):
  """Tokenizes a list of text strings using the given BERT tokenizer.

  Args:
    tokenizer: A BERT tokenizer.
    texts: A list of text strings.

  Returns:
    A list of tokenized sentences, where each sentence is a list of token IDs.
  """

  tokenized_texts = []
  for text in texts:
    # Fix the error by removing the `return_tensors='pt'` argument.
    tokens = tokenizer(text, padding=True, truncation=True, max_length=128)['input_ids']
    tokenized_texts.append(tokens)

  return tokenized_texts
@app.post('/predict')


async def predict(UserInput:UserInput):

    #pre=json.loads(UserInput.prediction_input)
    #pre =json.dumps(UserInput.prediction_input)
    #new_arr = np.array(UserInput.prediction_input)
    
    k=train.le
    
    text_p=[]
   
    text_p.append(UserInput.prediction_input)
    
    prediction_input = tokenize_data(tokenizer, text_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], 16)
    prediction=model.predict(prediction_input)
    output=prediction.argmax()
    respones_tag=k.inverse_transform([output])[0]

        
  
    for intent  in   data["intents"]:
            if intent['tag']==respones_tag:
                responses=intent['responses']
    o=(random.choice(responses))
     
    
      
        
            
    

    return {o}