from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.docstore.document import Document
from langchain_classic.text_splitter import TokenTextSplitter
from langchain_classic.chat_models import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.chains import retrieval_qa
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from src.promt import * # this imports prompt_template and refine_template


# Gemini Authantication
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Function to process the PDF file and return documents for question generation and answer generation
def file_processing(file_path):

    # Load the PDF document
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen= '' 

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen= TokenTextSplitter(
    model_name="gpt-3.5-turbo",
    chunk_size=10000,
    chunk_overlap=200,
)
    # Split the text into chunks
    chunk_quiz_gen = splitter_ques_gen.split_text(question_gen)

    # Create Document objects for each chunk
    documents_quiz_gen = [Document(page_content=chunk) for chunk in chunk_quiz_gen]

    splitter_ans_gen= TokenTextSplitter(
    model_name="gpt-3.5-turbo",
    chunk_size=1000,
    chunk_overlap=100,
)   

    # Split the text into smaller chunks for answer generation
    documents_answer_gen = splitter_ans_gen.split_documents(documents_quiz_gen)

    return documents_quiz_gen, documents_answer_gen


def llm_pipeline(file_path):

    document_quiz_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)
    
    PROMT_QUESTION = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMT_QUESTION = PromptTemplate(template=refine_template, 
                                           input_variables=["existing_answer", "text"])
    
    # Create the question generation chain
    # "verbose=True" will print out the intermediate steps of the chain
    # "refine" indicates that we are using the refine chain type, that means we will first generate an initial answer and then refine it with more context 
    question_gen_chain = load_summarize_chain(llm_ques_gen_pipeline,
                                                chain_type="refine",
                                                verbose=True,
                                                question_prompt=PROMT_QUESTION,
                                                refine_prompt=REFINE_PROMT_QUESTION)
    
    # Generate questions
    questions = question_gen_chain.run(document_quiz_gen)

    # Create embeddings for the answer generation
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-2.5-flash-lite")

    # Create a vector store from the documents for answer generation
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Create the retrieval QA chain for answer generation
    llm_ans_gen_pipeline = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3
    )

    quiz_list = questions.split("\n")
    filtered_quiz_list = [element for element in quiz_list if element.endswith("?") or element.endswith(".")]


    answre_genaration_chain = retrieval_qa.RetrievalQA.from_chain_type(
        llm=llm_ans_gen_pipeline,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
    )

    return filtered_quiz_list, answre_genaration_chain
    