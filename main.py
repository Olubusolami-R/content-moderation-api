from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Content Moderation API")

# Load model once when server starts
classifier = pipeline("text-classification",
                      model="unitary/toxic-bert",
                      top_k=None)


class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Content Moderation API is running"}


@app.post("/moderate")
def moderate_text(input: TextInput):
    # Get prediction
    result = classifier(input.text)[0]

    # Find highest scoring label
    top_prediction = max(result, key=lambda x: x['score'])

    return {
        "text": input.text,
        "prediction": top_prediction['label'],
        "confidence": top_prediction['score'],
        "all_scores": result
    }


@app.get("/health")
def health():
    return {"status": "healthy"}