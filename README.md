# 🎓 ML Model for Academic Purposes

An AI-powered academic chatbot designed to assist students with academic queries using NLP and sentence similarity techniques.

## 📋 Table of Contents

- [Introduction](#introduction)
- [✨ Features](#features)
- [⚙️ Installation](#installation)
- [🚀 Usage](#usage)
- [📐 Model and Methodology](#model-and-methodology)
- [💻 Technologies Used](#technologies-used)
- [📁 Project Structure](#project-structure)
- [🔮 Future Work](#future-work)
- [📜 License](#license)

## Introduction

The **ML Model for Academic Purposes** chatbot is designed to provide real-time answers to students’ frequently asked academic questions. This project uses Sentence-BERT embeddings and cosine similarity to deliver the most relevant answer to a user's query.

## ✨ Features

- **NLP-Powered**: Understands academic questions through Sentence-BERT embeddings.
- **High Precision Matching**: Finds the best answer using cosine similarity.
- **User-Friendly UI**: Responsive web interface with Flask and HTML/CSS.
- **Error Handling**: Friendly messages when something goes wrong.

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher 🐍
- Libraries listed in `requirements.txt`

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/gyerra/AIML-PROJECT-2320040080.git
   cd AIML-PROJECT-2320040080

2.**Install dependencies**
  pip install -r requirements.txt

3.**Dataset**
Place dataset.xlsx in the root directory

4.**Run the app**
  python app.py
_Open the chatbot at http://127.0.0.1:5000/ in your browser._

🚀 **Usage**
Type your question in the chatbot's input field.
Hit "Ask" to get your answer.
View the response, or error if no match was found.

📐 **Model and Methodology**
Sentence-BERT (all-MiniLM-L6-v2): Embeds student queries and compares them with each question in the dataset.
Cosine Similarity: Measures similarity between the user query and dataset queries to find the best answer.
Flask Backend: Routes user queries to the model and returns the results.

💻 **Technologies Used**
Python: Core programming language.
Pandas: Data processing.
Sentence-Transformers: For Sentence-BERT embeddings.
Flask: Web app framework.
HTML/CSS: Frontend development.

🔮 **Future Work**
Improve Accuracy: Test other models to enhance response accuracy.
New Features: Add voice input, multilingual support, etc.
Deployment: Make the chatbot publicly accessible for educational use.

**📜License**
Licensed under the MIT License. See the LICENSE file for details.

