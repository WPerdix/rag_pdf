import pymupdf
import faiss
import numpy as np
import ollama

from llama_index.llms.ollama import Ollama


class VectorStore():
    def __init__(self, pdf_path):
        
        self.llm = Ollama(model='mistral')
        
        self.paragraphs = []

        with pymupdf.open(pdf_path) as file:
            for page in file:
                self.paragraphs.append(page.get_text('text'))
        text_embeddings = np.array([self.get_text_embedding(chunk) for chunk in self.paragraphs])
        
        d = text_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        
        for i in range(text_embeddings.shape[0]):
            text_embeddings[i] /= np.linalg.norm(text_embeddings[i])
        
        self.index.add(text_embeddings)
        
    def get_text_embedding(self, prompt):
        return ollama.embed(model='nomic-embed-text', input=prompt).embeddings[0]
        
    def query(self, question):
        embedding = np.array([self.get_text_embedding(question)])
        
        D, I = self.index.search(embedding, k=2) # distance, index
        retrieved_chunk = [self.paragraphs[i] for i in I.tolist()[0]]
        
        prompt = f"""
            Context information is below.
            ---------------------
            {retrieved_chunk}
            ---------------------
            Given the context information and not prior knowledge, answer the query.
            Query: {question}
            Answer:
            """
            
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]
 

path = './rulebook-zombicide-season-1.pdf'

vectorstore = VectorStore(path)

response = vectorstore.query("Tell me about this board game. What are the different stages in a player's turn for this board game?")
print(response)

