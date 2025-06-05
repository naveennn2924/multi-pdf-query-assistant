# PDF Query Streamlit Application

This application allows users to upload multiple PDF documents, extract their content, and ask questions based on the information from the PDFs. The app uses machine learning models to retrieve relevant answers from the documents.

## Features
- Upload and process multiple PDF files.
- Ask questions related to the content of the uploaded PDFs.
- Answer extraction using a language model that processes the PDF content.
- A user-friendly interface with a modern and clean design.

## Requirements

To run this app locally, you'll need to install the following dependencies:

- Python 3.10
- Streamlit
- PyPDF2
- Langchain
- HuggingFaceHub
- FAISS
- dotenv

## Installation

1. **Clone this repository or download the script.**
   
   If you haven't already cloned this repository, you can do it using the following command:


   https://github.com/naveennn2924/multi-pdf-query-assistant.git


## Usage
Run the application:

Once you have installed the necessary libraries and set up the environment, run the Streamlit application using:
< streamlit run app.py >
Upload PDFs:

Go to the Streamlit interface in your web browser.
## Upload your PDF files using the file uploader.
The app will process the PDFs, extract their content, and store it in a vector store for later use.
## Ask Questions:

After uploading and processing the PDFs, enter your question in the input field.
The system will retrieve relevant answers based on the content extracted from the PDFs and display them on the screen.
## How It Works
PDF Text Extraction: The app extracts text from uploaded PDF files using PyPDF2.
Text Chunking: The extracted text is split into smaller chunks using RecursiveCharacterTextSplitter to make it manageable for the model.
Vector Store: The app uses the FAISS vector store to store the embeddings of the text chunks for efficient similarity search.
Question Answering: Once the text is processed, the app allows users to input questions. The relevant documents are retrieved from the vector store, and the answer is generated using a language model (HuggingFaceHub).
## Customization
You can modify the chat_llm model by changing the repo_id in the HuggingFaceHub to another pre-trained model from HuggingFace.
Adjust the chunk size and overlap in the get_text_chunks() function if the documents are very large.
## App Design
The application uses Streamlit for the front-end, providing a simple and easy-to-use interface.
Custom CSS styles are added to improve the overall appearance of the app, including styling for input fields, buttons, and response boxes.
The app supports multiple file uploads and displays real-time updates as PDFs are processed.
## Troubleshooting
If you encounter issues with dependencies: Ensure that all dependencies are correctly installed by checking the versions in requirements.txt or using a virtual environment.
If the PDFs are not processed correctly: Check that the PDF files are not encrypted or corrupted.
Slow performance: This app may take time to process large documents, especially when many PDFs are uploaded.
