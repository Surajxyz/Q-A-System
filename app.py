from flask import Flask, request, render_template,session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from werkzeug.utils import secure_filename
import os
import json


app = Flask(__name__,template_folder='Template')

j=0

app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'C:\\Users\\USER\\Chat bot\\PDF_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['pdf'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["GET","POST"])
def login():
    return render_template("index2.html")

@app.route('/upload', methods=["GET",'POST'])
def upload():
    session["context_question"]=[]
    directory_path1= "C:\\Users\\USER\\Chat bot\\PDF_files"
    files1=os.listdir(directory_path1)
    for file in files1:
        file_path=os.path.join(directory_path1,file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
    files = request.files['pdfFile']
    filename = secure_filename(files.filename)
    files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if allowed_file(filename):
        question=request.form.get("textInput")
        loader = PyPDFLoader(f"{UPLOAD_FOLDER}\\{filename}")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0,separators=".")
        docs = text_splitter.split_documents(pages)
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"

        # Create a dictionary with model configuration options, specifying to use the CPU for computations
        model_kwargs = {'device':'cpu'}

        # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
        encode_kwargs = {'normalize_embeddings': False}

        # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
        )
        db = FAISS.from_documents(docs, embeddings)
        if request.form.get("btnTask1")=="click1":
            searchDocs = db.similarity_search(question)
            context=searchDocs[0].page_content
            context_question = [(doc.page_content, question) for doc in searchDocs]
            session["context_question"]=json.dumps(context_question)
            model_name = "Intel/dynamic_tinybert"

    # Load the tokenizer associated with the specified model
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512,max_token_length=512)

            # Define a question-answering pipeline using the model and tokenizer
            question_answerer = pipeline(
                "question-answering", 
                model=model_name, 
                tokenizer=tokenizer,
                return_tensors='pt',
                max_answer_len=512
            )
            answer=question_answerer(question=question, context=context)
        return render_template("index1.html",answer=f"answer: {answer['answer']}",context=f"context: {context}",question=f"question: {question}")
    else:
        return render_template("index1.html",answer="Ender a PDF file only")

                

@app.route('/upload/feedback', methods=["GET",'POST'])
def feedback():
    context_question = json.loads(session['context_question'])
    global j 
    if request.form.get("btnTask3")=="click3":
        
        j+=1
        if j <len(context_question):
           
            model_name = "Intel/dynamic_tinybert"


            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512,max_token_length=512)

       
            question_answerer = pipeline(
                "question-answering", 
                model=model_name, 
                tokenizer=tokenizer,
                return_tensors='pt',
                max_answer_len=512
                )
            answer=question_answerer(question=context_question[j][1], context=context_question[j][0])
            return render_template("index1.html",answer=f"answer: {answer['answer']}",context=f"context: {context_question[j][0]}",question=f"question: {context_question[j][1]}")
        else:
            j=0
            return render_template("index1.html",answer="Feedback given limit exceed, please restart")

    elif request.form.get("btnTask2")=="click2":
        return render_template("index1.html",answer="Thank you for your positive feedback")    


if __name__ == '__main__':
    app.run(debug=True)