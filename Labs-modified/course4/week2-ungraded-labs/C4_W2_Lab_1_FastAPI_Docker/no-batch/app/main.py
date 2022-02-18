import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Predicting Wine Class")

# Represents a particular wine (or datapoint)
class Wine(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.on_event("startup") #ensure that the function is run at the startup of the server
def load_clf():
    # Load classifier from pickle file
    with open("/app/wine.pkl", "rb") as file:
        global clf #make global so other functions can access it
        clf = pickle.load(file) #load pre-trained model saved in the app/wine.pkl file


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"


@app.post("/predict") # the path in bracket like "/predict" are endpoint path of the server, e.g. '/' is the root path of the server 
def predict(wine: Wine):
    #convert information within Wine object into a numpy array to feed to model.predict function.
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash,
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )

    #model.predict to make prediction using loaded model
    pred = clf.predict(data_point).tolist() #Notice that the prediction must be casted into a list using the tolist method.
    pred = pred[0]
    print(pred)
    return {"Prediction": pred} #return a dictionary (which FastAPI will convert into JSON) containing the prediction.
