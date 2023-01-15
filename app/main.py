from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from fastapi import FastAPI, Form
from src.models import train_model



tokenizer = AutoTokenizer.from_pretrained("lucixls/models")
model = AutoModelForSequenceClassification.from_pretrained("lucixls/models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI()

@app.get("/")
def read_root():
   return {"Hello": "World"}



@app.post("/predict")
async def predict(text: str = Form(...)):
    # Encode text input
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    # Perform inference
    output = model(input_ids)[0]
    _, preds = torch.max(output, dim=1)
    # Convert prediction to string
    label = preds.item()
    return {"label": label}