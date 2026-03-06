import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_utils import load_clean_data
from logger import logger

class MarketingRAG:
    def __init__(self):
        logger.info("Initializing Marketing RAG System")
        
        # Load the cleaned dataset
        self.df = load_clean_data()
        
        # Ensure product_name exists and handle potential missing values
        if 'product_name' not in self.df.columns:
            self.df['product_name'] = "Unknown Product"
        else:
            self.df['product_name'] = self.df['product_name'].fillna("Unknown Product")

        # Create a rich text representation of each product for the knowledge base
        logger.info("Preparing knowledge base documents")
        self.df['context'] = self.df.apply(
            lambda row: f"Product: {row['product_name']}, Category: {row['category_main']}, "
                        f"Price: {row['actual_price']}, Rating: {row['rating']} stars.",
            axis=1
        )
        self.documents = self.df['context'].tolist()

        #Embedding Model
        logger.info("Loading embedding model")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Build FAISS Vector Index
        logger.info("Building FAISS index")
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        
        # Initialize FAISS with the embedding dimension size
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # Initialize Generative LLM
        logger.info("Loading generative LLM")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        logger.info("Marketing RAG System is ready.")

    def retrieve(self, query, top_k=3):
        # Convert user query to vector
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search the FAISS index for the closest matches
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Fetch the actual text for the matched indices
        retrieved_docs = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        return retrieved_docs

    def generate(self, query):
        logger.info(f"Generating RAG response for query: {query}")
        
        # Retrieve relevant context
        retrieved_docs = self.retrieve(query, top_k=3)
        context = " | ".join(retrieved_docs)

        # Construct prompt grounding the LLM in the retrieved data
        prompt = (
            f"Answer the user's question using ONLY the provided context.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Generate the response explicitly
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"LLM Generation Error: {str(e)}")
            return "I'm sorry, I encountered an error while trying to generate an answer."