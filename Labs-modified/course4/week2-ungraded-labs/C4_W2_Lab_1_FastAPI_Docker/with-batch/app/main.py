import pickle
import numpy as np
from fastapi import FastAPI

#In this lab, we need to handle prediction by "batches". So we need to import the following two extra modules
#These are Lists from typing and conlist from pydantic. Remember that REST does not support objects like numpy arrays so you need to serialize this kind of data into lists instead
from typing import List
from pydantic import BaseModel, conlist


app = FastAPI(title="Predicting Wine Class with batching")

# Represents a batch of wines
class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]
    # Note: The "con" prefix stands for constrained, so this is a constrained list. 
    # This type allows you to select the type of the items within the list and also the maximum and minimum number of items. 
    # In this case your model was trained using 13 features so each data point should be of size 13 (example see lab part 1 no-batch's main.py)


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("../app/wine.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. This new version allows for batching. Now head over to http://localhost:81/docs"


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches #return a list object based on "Class Wine" definition
    np_batches = np.array(batches) #convert list to np.array to feed to model

    pred = clf.predict(np_batches).tolist() #convert the predictions to a list to make them REST-compatible
    return {"Prediction": pred}
