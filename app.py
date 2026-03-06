from fastapi import FastAPI
import joblib
import numpy as np
from logger import logger
from rag import MarketingRAG
from monitoring import ModelMonitor

app = FastAPI()

# Load model and encoder
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
rag_system = MarketingRAG()
monitor = ModelMonitor()


@app.get("/")
def home():
    return {"message": "Marketing AI Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict_discount")
def predict_discount(data: dict):

    try:

        logger.info("Prediction request received")

        category = data["category"]
        actual_price = float(data["actual_price"])
        rating = float(data["rating"])
        rating_count = int(data["rating_count"])

        # Encode category
        if category not in label_encoder.classes_:
            category_encoded = -1
        else:
            category_encoded = label_encoder.transform([category])[0]

        input_data = np.array([[actual_price,
                                rating,
                                rating_count,
                                category_encoded]])

        prediction = model.predict(input_data)

        logger.info(f"Prediction generated: {prediction[0]}")

        monitor.check_all_features({
            "actual_price": actual_price,
            "rating": rating,
            "rating_count": rating_count
        })

        return {"predicted_discount_percent": float(prediction[0])}

    except Exception as e:

        logger.error(f"Prediction error: {str(e)}")

        return {"error": str(e)}


@app.post("/answer_question")
def answer_question(data: dict):
    try:
        #Handle a list of queries
        if "queries" in data and isinstance(data["queries"], list):
            queries = data["queries"]
            logger.info(f"RAG multiple queries received: {len(queries)} questions")
            
            results = []
            for q in queries:
                ans = rag_system.generate(q)
                results.append({"query": q, "answer": ans})
                
            return {"results": results}
            
        #Handle a single query
        elif "query" in data:
            query = data["query"]
            logger.info(f"RAG single query received: {query}")
            
            response = rag_system.generate(query)
            return {"answer": response}
            
        #Handle bad payloads
        else:
            logger.warning("Invalid RAG payload received.")
            return {"error": "Invalid payload. Please provide 'query' (string) or 'queries' (list of strings)."}

    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        return {"error": str(e)}