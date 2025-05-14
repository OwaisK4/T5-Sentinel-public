import numpy
from pathlib import Path
import evaluator.models.t5_sentinel.t5_pipeline as pipeline
from fastapi import FastAPI, Request

labels = ["Human", "ChatGPT", "Claude", "Gemini"]

app = FastAPI()
model = pipeline.ExecuteT5(Path("./data/checkpoint/T5Sentinel.0613.pt"))


@app.get("/")
def read_root():
    return {"message": "Hello from T5-Multi!"}


@app.post("/predict")
async def predict(request: Request):
    # dummy response using input
    data = await request.json()
    output = model(data)
    result = pipeline.T5PredictToLogits()(output)
    prediction = numpy.argmax(result["data"]).item()

    return {
        "prediction": labels[prediction],
        "score": str(result["data"][numpy.argmax(result["data"])]),
        "per_class": result["data"],
    }
