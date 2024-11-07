import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, render_template, jsonify
import logging
import traceback
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create dataset directly in the code
data = {
    'Student Query': [
        'When are the final exams scheduled?',
        'What are the course details for ECE 101?',
        'How can I access my previous semester results?',
        'Where can I find the library hours?',
        'What is the prerequisite for ECE 202?',
        'Who is the instructor for CS101?',
        'Can I change my major?',
        'When do classes start for the next semester?',
        'How do I apply for graduation?',
        'How can I join a student club?'
    ],
    'Answer': [
        'Final exams are scheduled from December 10th to December 20th.',
        'ECE 101 is an introductory course on electronics. It covers basic circuit theory and electronic devices. Instructor: Prof. Srinivasa Rao.',
        'You can access your previous semester results via the student portal under the "Grades" section.',
        'The library is open from 8 AM to 9 PM on weekdays, and from 10 AM to 2 PM on weekends.',
        'The prerequisite for ECE 202 is ECE 101.',
        'The instructor for CS101 is Dr. Priya Mehta.',
        'Yes, you can change your major by submitting a request to the academic office. The deadline for major change requests is ecember 15th.',
        'Classes for the next semester start on January 5th.',
        'You can apply for graduation by filling out the graduation application form on the student portal. The deadline for applications is April 1st.',
        'You can join a student club by attending the club fairs held at the start of each semester or by contacting the club coordinators directly.'
    ]
}


df = pd.DataFrame(data)
logging.info(f"Dataset created successfully. Shape: {df.shape}")
logging.info(f"Columns: {df.columns.tolist()}")

# Load the Sentence-BERT model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence-BERT model loaded successfully.")

    # Encode all queries in the dataset
    df['Query Embedding'] = df['Student Query'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    logging.info("Embeddings for all queries created successfully.")

except Exception as e:
    logging.error(f"Error during model loading or embedding: {str(e)}")
    raise

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form['query']
        logging.info(f"Received query: {user_query}")

        if not user_query:
            logging.error("Empty query received.")
            return jsonify({'error': 'Empty query received'}), 400

        # Encode the user query
        user_query_embedding = model.encode(user_query, convert_to_tensor=True)

        # Compute cosine similarity with all query embeddings in the dataset
        similarities = [util.cos_sim(user_query_embedding, query_emb).item() for query_emb in df['Query Embedding']]
        
        # Find the index of the highest similarity score
        nearest_query_index = similarities.index(max(similarities))
        answer = df.iloc[nearest_query_index]['Answer']

        logging.info(f"Found answer: {answer}")
        return jsonify({'response': answer})

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)