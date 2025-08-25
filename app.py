import os
import uuid
import pymupdf
import faiss
import numpy as np
import ollama

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from llama_index.llms.ollama import Ollama

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store vector stores for each session
vector_stores = {}

class VectorStore():
    def __init__(self, pdf_path):
        self.llm = Ollama(model='mistral')
        self.paragraphs = []

        with pymupdf.open(pdf_path) as file:
            for page in file:
                text = page.get_text('text').strip()
                if text:  # Only add non-empty pages
                    self.paragraphs.append(text)
        
        if not self.paragraphs:
            raise ValueError("No text content found in PDF")
        
        text_embeddings = np.array([self.get_text_embedding(chunk) for chunk in self.paragraphs])
        
        d = text_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        
        # Normalize embeddings
        for i in range(text_embeddings.shape[0]):
            text_embeddings[i] /= np.linalg.norm(text_embeddings[i])
        
        self.index.add(text_embeddings)
        
    def get_text_embedding(self, prompt):
        return ollama.embed(model='nomic-embed-text', input=prompt).embeddings[0]
        
    def query(self, question):
        try:
            embedding = np.array([self.get_text_embedding(question)])
            for i in range(embedding.shape[0]):
                embedding[i] /= np.linalg.norm(embedding[i])
            
            D, I = self.index.search(embedding, k=3)  # distance, index
            retrieved_chunks = [self.paragraphs[i] for i in I.tolist()[0]]
            
            context = "\n".join(retrieved_chunks)
            
            prompt = f"""Context information is below.
                ---------------------
                {context}
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: {question}
                Answer:"""
                
            response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response["message"]["content"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(filepath)
            
            # Create vector store
            vector_store = VectorStore(filepath)
            vector_stores[session_id] = {
                'store': vector_store,
                'filename': filename,
                'filepath': filepath
            }
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'filename': filename,
                'message': 'PDF uploaded and processed successfully!'
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question')
        
        if not session_id or not question:
            return jsonify({'error': 'Missing session_id or question'}), 400
        
        if session_id not in vector_stores:
            return jsonify({'error': 'Session not found. Please upload a PDF first.'}), 404
        
        vector_store = vector_stores[session_id]['store']
        answer = vector_store.query(question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing chat: {str(e)}'}), 500

@app.route('/sessions', methods=['GET'])
def get_sessions():
    sessions = []
    for session_id, data in vector_stores.items():
        sessions.append({
            'session_id': session_id,
            'filename': data['filename']
        })
    return jsonify({'sessions': sessions})

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        if session_id in vector_stores:
            # Delete the uploaded file
            filepath = vector_stores[session_id]['filepath']
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove from memory
            del vector_stores[session_id]
            
            return jsonify({'success': True, 'message': 'Session deleted successfully'})
        else:
            return jsonify({'error': 'Session not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Error deleting session: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    