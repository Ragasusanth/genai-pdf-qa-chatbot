## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

Large documents such as PDFs contain a vast amount of information, making it difficult for users to quickly find specific answers. Manually searching through these documents is time-consuming and inefficient. To address this issue, a chatbot system can be developed that reads the content of a PDF file and answers user questions based on that information.

Using LangChain, the system processes the PDF, converts the text into embeddings, stores them in a vector database, and retrieves the most relevant sections when a user asks a question. The chatbot then uses a language model to generate accurate answers based on the retrieved content.
### DESIGN STEPS:

### STEP 1: Load and Process the PDF Document
Import the required LangChain libraries and load the PDF file using a document loader. Split the extracted text into smaller chunks to make it easier for the model to process.

### STEP 2: Create Embeddings and Vector Database
Convert the text chunks into embeddings using an embedding model and store them in a vector database (such as FAISS or Chroma) to enable efficient similarity search.

### STEP 3: Build the Question-Answering Chatbot
Use a retrieval-based QA chain in LangChain that retrieves relevant text from the vector database and sends it to the language model to generate accurate answers to user queries.

### PROGRAM:
```
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Load OpenAI API key
openai_api_key = os.environ['OPENAI_API_KEY']

def load_pdf_to_db(file_path):
    # Load the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # Set retriever to fetch relevant document chunks
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
def create_conversational_chain(retriever):
    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define the conversational retrieval chain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),  # Using OpenAI Chat model
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
if __name__ == "__main__":
    # Load the PDF file
    pdf_file_path = "DOC1.pdf"  # Replace with your file path
    retriever = load_pdf_to_db(pdf_file_path)
    
    # Create chatbot chain
    chatbot = create_conversational_chain(retriever)
    
    # Start a conversation
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])
```
### OUTPUT:
![image alt](https://github.com/Ragasusanth/genai-pdf-qa-chatbot/blob/0aba9fee4210378763df2b63c1e59050305fb695/OUTPUT%20FOR%20EX03.png)
### RESULT:
The PDF-based question-answering chatbot was successfully developed using LangChain. The system was able to process the PDF document, retrieve relevant information, and generate accurate answers to user queries based on the document’s content.
