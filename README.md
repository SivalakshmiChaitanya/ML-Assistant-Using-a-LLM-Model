# Marketing Data Intelligence API

An end-to-end machine learning system designed to enhance data-driven decision-making. This project features a containerized FastAPI application hosting two distinct AI systems:
1. **Predictive Model:** A Random Forest Regressor trained to predict optimal product discounts based on historical data.
2. **Generative AI (RAG):** An LLM-powered assistant (`google/flan-t5-small`) using vector search (`FAISS` & `all-MiniLM-L6-v2`) to answer natural language questions about the product catalog.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
* **Python 3.10+** (if running locally)
* **Docker** (if running via container)

---

## 🛠️ Step 1: Setup & Installation

**1. Navigate to the project directory:**
Open your terminal or command prompt and `cd` into the folder containing your code.

**2. Install the required Python dependencies:**
pip install -r requirements.txt

---

## 🧠 Step 2: Train the Predictive Model

Before running the API for the first time, you must train the Random Forest model and generate the necessary baseline statistics for drift monitoring.

**Run the training script:**
python train.py

*This will process `amazon.csv` and generate three new files: `model.pkl`, `label_encoder.pkl`, and `training_stats.json`.*

---

## 🚀 Step 3: Run the Application

You can run the API either locally on your machine or inside a Docker container.

### Option A: Run Locally (Uvicorn)
Start the FastAPI server directly:
uvicorn app:app --host 0.0.0.0 --port 8000

*(Note: The first time you start the server, it may take a minute to download the Hugging Face LLM models to your machine).*

### Option B: Run via Docker (Recommended for Production)
**1. Build the Docker image:**
docker build -t MLAssignment .

**2. Run the Docker container:**
docker run -p 8000:8000 MLAssignment

---

## 📡 Step 4: API Usage & Endpoints

Once the server is running (usually at `http://127.0.0.1:8000`), you can test the endpoints.

### 1. Health Check
* **Endpoint:** `GET /health`
* **Description:** Verifies the API is online.

### 2. Predict Discount
* **Endpoint:** `POST /predict_discount`
* **Description:** Predicts the optimal discount percentage for a product based on its features.
* **JSON Payload:**
  {
    "category": "Electronics",
    "actual_price": 1500.0,
    "rating": 4.2,
    "rating_count": 1250
  }

### 3. Answer Question (RAG)
* **Endpoint:** `POST /answer_question`
* **Description:** Queries the RAG knowledge base. It accepts either a single `"query"` or a list of `"queries"` for batch processing.
* **JSON Payload (Batch Example):**
  {
    "queries": [
      "What is the most expensive product?",
      "What is the average rating?"
    ]
  }

---

## 📊 Step 5: Monitoring & Logging

* **API Traffic:** All requests and errors are logged to the `app.log` file in the project directory, as well as streamed to the console.
* **Data Drift Detection:** The API automatically monitors incoming `/predict_discount` requests. If the input data deviates significantly (z-score > 3) from the original training data, a `WARNING: Possible drift detected` message will be printed directly in the server console.

---

## 🧪 Step 6: Running Tests

This project includes an automated integration test suite to verify all endpoints are functioning correctly. 

To run the tests, execute the following command in your terminal:
pytest test_app.py -v