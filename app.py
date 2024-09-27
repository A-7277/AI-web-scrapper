import streamlit as st 
import os
import google.generativeai as genai
import urllib3
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 


load_dotenv()
generation_config = {
  "temperature": 2,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

genai.configure(api_key=os.getenv('api_key'))


urls=[]

with st.sidebar:
  st.subheader('URL web scrapper Q/A')
  n = st.slider("Select the number of urls to be searched", 1, 10, 1)
  st.subheader('Enter your URLS below!')
  for i in  range  (n):
    urls.append(st.text_input(f'URL-{i+1}',key=i))
  # st.text(f'{len(urls[0])}{urls}')
if len(urls[0]) > 0:
  loader=UnstructuredURLLoader(urls=urls)
  data=loader.load()
  splitter=RecursiveCharacterTextSplitter(separators=['\n',' '],chunk_size=1000,chunk_overlap=200)
  chunks=splitter.split_documents(data)
  
  model_name = "sentence-transformers/all-mpnet-base-v2"
  model_kwargs = {'device': 'cpu'}
  encode_kwargs = {'normalize_embeddings': False}
  hf = HuggingFaceEmbeddings(
      model_name=model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
  )

  vectorIndex=FAISS.from_documents(chunks,hf)
  
  retriever=vectorIndex.as_retriever()
 
   
  messages = st.container( border=True)
  if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    
    para=retriever.get_relevant_documents(prompt) 
    # st.text(para)
    llm=genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
      system_instruction=f"find answer of this question- {prompt} from this paragarph {para}",
  )  
    
    messages.chat_message("assistant").write(f"Echo: {(llm.generate_content(prompt)).text}") 
        


