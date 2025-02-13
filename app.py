from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import difflib
import gzip
import os

app = FastAPI()

# Load the medicine dictionary and similarity matrix with error handling
try:
    medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
    medicines = pd.DataFrame(medicines_dict)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="medicine_dict.pkl file not found")

try:
    with gzip.open("similarity_compressed.pkl.gz", "rb") as f:
        similarity = pickle.load(f)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="similarity_compressed.pkl.gz file not found")

class MedicineRequest(BaseModel):
    medicine_name: str

def find_closest_medicine(query):
    all_medicines = medicines['Drug_Name'].tolist()
    closest_matches = difflib.get_close_matches(query, all_medicines, n=1, cutoff=0.5)
    return closest_matches[0] if closest_matches else None

def recommend(medicine):
    # Find the index of the closest medicine in the DataFrame
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    # Retrieve the similarity distances for the found medicine
    distances = similarity[medicine_index]
    # Sort the distances and get the top 5 recommendations (excluding the medicine itself)
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_medicines = [medicines.iloc[i[0]].Drug_Name for i in medicines_list]
    return recommended_medicines

@app.post("/recommend")
def get_recommendations(request: MedicineRequest):
    # Find the closest match for the requested medicine
    closest_medicine = find_closest_medicine(request.medicine_name)
    
    if not closest_medicine:
        raise HTTPException(status_code=404, detail="No similar medicine found")
    
    # Get recommendations based on the closest match
    recommendations = recommend(closest_medicine)
    return {"medicine_searched": closest_medicine, "recommended_medicines": recommendations}
