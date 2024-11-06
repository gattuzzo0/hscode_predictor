"""
HS Code Prediction App

This script is a Streamlit application designed to assist companies in identifying and classifying products accurately using 
the Harmonized System (HS) code based on product descriptions. Key features include:

- User inputs a product description, focusing on characteristics such as type, use, material, and dimensions.
- The script uses a combination of custom CSS styles for an enhanced user interface.
- Natural Language Processing (NLP) pipelines powered by LangChain and Hugging Face embeddings are used for keyword extraction and translation (if needed).
- The retrieval system, using Chroma as a vector store, retrieves HS codes based on similarity to the input description.
- Outputs the most relevant HS code with a high relevance score, along with supporting source documents for verification.

Main Components:
- Keyword extraction using language model (LLM).
- Translation prompt (for handling Spanish and English inputs).
- HS code retrieval based on contextual matching from the shipping manifest description.
- Streamlit layout for an intuitive and visually appealing interface.
"""



import os, re
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from transformers import AutoModel
import warnings
#import torch
from os.path import dirname, abspath
import torch

# Suppress all warnings
warnings.filterwarnings("ignore")


def extract_keywords_from_query(query):
    # Step 1: Define the keyword extraction prompt and LLMChain
    # Prompt to ask the LLM to extract relevant keywords
    keyword_extraction_template = (
        """You're a freight inspector, your job is to identify what are the most relevant keywords to identify the proper HS Code later:
            Text: {query}
            Respond ONLY with the keywords (ENGLISH or Spanish depending on your input) you think are most relevant to identify the content of that container, 
            as a whole sentence, NO OTHER TEXT"""
    )

    keyword_prompt = PromptTemplate(template=keyword_extraction_template, input_variables=["query"])

    # Initialize LLM for keyword extraction
    keyword_chain = LLMChain(llm=llm_model, prompt=keyword_prompt)

    # Extract keywords by running the LLM chain
    keywords_response = keyword_chain.run({"query": query})
    return keywords_response

def translate_query(query):
    # Step 1: Define the keyword extraction prompt and LLMChain
    # Prompt to ask the LLM to extract relevant keywords
    keyword_extraction_template = (
        """You're a Spanish-to-English translator with relevant experience in the freight business. Your job is to translate the query to English:
            Text: {query}
            Respond ONLY with the translation in English, as a whole sentence, NO OTHER TEXT or polite introduction"""
    )

    keyword_prompt = PromptTemplate(template=keyword_extraction_template, input_variables=["query"])

    # Initialize LLM for keyword extraction
    keyword_chain = LLMChain(llm=llm_model, prompt=keyword_prompt)

    # Extract keywords by running the LLM chain
    keywords_response = keyword_chain.run({"query": query})
    return keywords_response

def extract_keywords_from_query_in_english(query):
    # Step 1: Define the keyword extraction prompt and LLMChain
    # Prompt to ask the LLM to extract relevant keywords
    keyword_extraction_template = (
        """You're a freight inspector, your job is to identify what are the most relevant keywords to identify the proper HS Code later:
            Text: {query}
            Respond ONLY with ENGLISH keywords you think are most relevant to identify the content of that container, 
            as a whole sentence, NO OTHER TEXT"""
    )

    keyword_prompt = PromptTemplate(template=keyword_extraction_template, input_variables=["query"])

    # Initialize LLM for keyword extraction
    keyword_chain = LLMChain(llm=llm_model, prompt=keyword_prompt)

    # Extract keywords by running the LLM chain
    keywords_response = keyword_chain.run({"query": query})
    return keywords_response

#In case GPU is needed to be activated
#print(f"CUDA Available: {torch.cuda.is_available()}")
#if torch.cuda.is_available():
#    print(f"Device Name: {torch.cuda.get_device_name(0)}")
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
#model_kwargs = {'device': device , 'trust_remote_code': True}

# Set hyperparameters
index_test_name = 'Alibaba'
LLM_model = 'Llama3.1'
temperature_parameter = 0
retriever_matches_k = 3
embed_model = 'Alibaba-NLP/gte-large-en-v1.5'

# Define paths and initialize models
script_dir = abspath(dirname(__file__))
persistent_dir = os.path.abspath(os.path.join(script_dir, '..', 'index', index_test_name))

model = AutoModel.from_pretrained(embed_model, trust_remote_code=True) 
model_name = embed_model
model_kwargs = {'device': 'cpu' , 'trust_remote_code': True}     #In case you need to enable GPU, comment this line
encode_kwargs = {'normalize_embeddings': True}

embedding_model_via_Transformers_class = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# loading the LLM model
llm_model = OllamaLLM(model=LLM_model,
                temperature=temperature_parameter,
                num_thread=8,
                )
# loading the vectorstore
vectorstore = Chroma(persist_directory=persistent_dir, embedding_function=embedding_model_via_Transformers_class)
# casting  the vectorstore as the retriever, taking the best 3 similarities
retriever = vectorstore.as_retriever(search_kwargs={"k":retriever_matches_k})

### Definining Prompt Templates to be used on main queries
template = """ You must strictly respond with the HS code that matches the information contained in the 'source' 
    field of the metadata from the source_documents provided. You are not allowed to respond with any HS code from your own knowledge. 
    Do not invent or guess. Only respond with one HS code from the provided metadata that has the highest relevance score from the retriever, 
    and it must be from the 'source' field.

    For example, if the source document mentions 'Potato starch' with 'source': '1108.13', your answer must be '1108.13'.

    Avoid responding with any other text. Respond only with 1 HS code, the one with better score from the retriever.

    If for some reason, you can't find any match, suggest 3 possible matches with their corresponding descriptions.

context:
{summaries}

Question:
{question}
"""

context = """As a logistics shipping arrival inspector, your primary responsibility is to inspect incoming shipments and accurately classify goods 
using the Harmonized System (HS) code based on the descriptions provided in the shipping manifests. You will thoroughly review the manifest details, 
including product type, material composition, function, and intended use, to determine the correct HS code. 

Your task is to:
Carefully read and analyze the product descriptions from the manifest.
Identify key characteristics of the goods, such as 
type (e.g., electronics, textiles, machinery), 
material (e.g., plastic, metal, organic), 
and usage (e.g., household, industrial, medical).
Use your knowledge of the HS code classification system to assign the most appropriate HS code for each product based on its description.
Ensure compliance with international trade regulations by selecting precise codes to avoid delays or penalties.
Remember to be thorough and accurate in your classification, as this impacts customs processing, tariffs, and legal requirements."""


# Define the LLM Q&A chain type
llm_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm_model,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,  # To get both the answer and source docs
    chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                #For some reason, "context" cant be used as input variable, it should be named as "summaries"
                input_variables=["question", "summaries"],
            ),
        },
)

# Load CSS file for nicer styling
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<h1 class="title-style">HS Code App</h1>', unsafe_allow_html=True)
# Use the custom text style
st.markdown(
    """
    <div class="text-style">
        Bienvenido a la HS Code APP, una herramienta que ayuda a las empresas a identificar y clasificar productos de manera más precisa.
    </div>
    """, 
    unsafe_allow_html=True
)

# Input description
st.markdown(
    "Sea lo más específico posible tratando de cubrir las características principales como:\n"
    "- **TIPO** (Herramienta, juguete, electrónicos...)\n"
    "- **USO (Doméstico, Industrial, Médico...)**\n"
    "- **MATERIAL (Cobre, Latón, Metal...)**\n"
    "- **DIMENSIONES (en cm, en pulgadas...)**\n\n"
)

# Caja de entrada de texto
query = st.text_area("Escriba aquí la descripción:")                     

# Run prediction when user submits input
if st.button("Predict HS Code"):
    if query:
        with st.spinner("Procesando..."):
            #Perform simple Chain Of Thoughts (Extract keywords, Translate, Extract translated keywords -> Run)
            # Extract keywords
            keywords = extract_keywords_from_query(query)
            # Translate if necessary
            translated_query = translate_query(keywords)
            # Extract translated keywords
            translated_keywords = extract_keywords_from_query_in_english(translated_query)

            # Run LLMChain
            result = llm_chain({"question": translated_keywords, "summaries": context})

            # Optional
            # Display what is looking for actually on the VectorDatabase
            #st.write(" On the backend, I'm looking for....")            
            st.markdown(
            f"""
            <div class="text-style">
                {translated_keywords}
            </div>
            """, 
            unsafe_allow_html=True
                    )
            # Display recommended HS Code on a nice HTML Title
            st.write("### HS Code apropiado:")
            st.markdown(
                        f"""
                        <div class="highlighted-selection">
                            {result["answer"]}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            # Display source documents if available on a nice HTML Table
            if "source_documents" in result:
                st.write("### HS Codes con mayor similitud:")
            
                # Start the HTML table structure with headers
                table_html = "<table><thead><tr><th>Code</th><th>Description</th></tr></thead><tbody>"
                for i in range(len(result["source_documents"])):
                    # Formatting metadata dictionary as a string
                    data = result["source_documents"][i].metadata
                    metadata_str =  re.sub(r'[^\d.]', '', data['source'])
                    content_str = result["source_documents"][i].page_content

                    # Add each document row to the table
                    table_html += f"<tr><td>{metadata_str}</td><td>{content_str}</td></tr>"
                
                # Close the HTML table structure
                table_html += "</tbody></table>"
                
                # Display the HTML table in Streamlit
                st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.warning("Error: Por favor, capture una descripcion mas detallada.")
