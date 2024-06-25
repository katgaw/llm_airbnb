import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
import openai
from dotenv import main

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
main.load_dotenv()

"""
We will load our environment variables here.
"""
openai.api_key=os.environ["OPENAI_API_KEY"]

# Model
openai_chat_model = ChatOpenAI(model="gpt-4o")

# upload embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}
Use the provide context to answer the provided user question. Only use the provided context to answer the question. If you do not know the answer, response with "I don't know"
"""

CONTEXT = """
You are an expert on Airbnb, be polite and answer all questions. This report on Airbnb 10k filings contains unstructured and structured tabular data, use both.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# ---- GLOBAL DECLARATIONS ---- #
# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
# upload file
#docs=TextLoader("./data/airbnb_10k_filings.txt").load()
docs = PyMuPDFLoader("airbnb_10k_filings.pdf").load()


def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

max_chunk_length = 0

for chunk in split_chunks:
    max_chunk_length = max(max_chunk_length, tiktoken_len(chunk.page_content))

# Embeddings and Vector store
# qdrant_vectorstore = FAISS.from_documents(
#     split_chunks,
#     embedding_model,
#     location=":memory:",
#     collection_name="airbnb 10k filings",
# )

if os.path.exists("./data/vectorstore"):
    vectorstore = FAISS.load_local(
        "./data/vectorstore", 
        embedding_model, 
        allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
    )
    retriever = vectorstore.as_retriever()
    print("Loaded Vectorstore")
else:
    print("Indexing Files")
    os.makedirs("./data/vectorstore", exist_ok=True)
    for i in range(0, len(split_chunks), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(split_chunks[i:i+32], embedding_model)
            continue
        vectorstore.add_documents(split_chunks[i:i+32])
    vectorstore.save_local("./data/vectorstore")


print("Loaded Vectorstore")

# Ste up ur retriever using LangChain
retriever = vectorstore.as_retriever()

@cl.on_chat_start
async def init():
    # -- Our RAG Chain -- #

    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    lcel_rag_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )
    # cl.user_session.set("retrieval_augmented_qa_chain", retrieval_augmented_qa_chain)
    
    # lcel_rag_chain = (
    #     {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    #     | rag_prompt | openai_chat_model
    # )
    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    # for chunk in await cl.make_async(lcel_rag_chain.stream)(
    #     {"question": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # await msg.send()
    print(msg)
    response = lcel_rag_chain.invoke({"question" : message.content})
    # lcel_rag_chain = cl.user_session.get("lcel_rag_chain")
    # res = lcel_rag_chain.invoke({"question":message.content})
    print(response["response"].content)
    await cl.Message(content=response["response"].content).send()
