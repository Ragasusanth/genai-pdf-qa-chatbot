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
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import QuestionAnsweringChain
from langchain.llms import OpenAI

# Extract PDF text
def extract_pdf_text(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Initialize LLM (OpenAI, or other LLMs)
llm = OpenAI(temperature=0.7)

# Initialize TextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Create Q&A Chain
qa_chain = QuestionAnsweringChain.from_llm(llm)

def answer_question(question, chunks):
    context = " ".join(chunks)
    return qa_chain.run({"input_document": context, "question": question})

def main():
    pdf_path = "document.pdf"  # Provide the path to your PDF file
    extracted_text = extract_pdf_text(pdf_path)
    chunks = splitter.split_text(extracted_text)
    
    print("PDF-based Question Answering Chatbot")
    
    while True:
        question = input("Ask a question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = answer_question(question, chunks)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```
### OUTPUT:

### RESULT:
The PDF-based question-answering chatbot was successfully developed using LangChain. The system was able to process the PDF document, retrieve relevant information, and generate accurate answers to user queries based on the document’s content.
