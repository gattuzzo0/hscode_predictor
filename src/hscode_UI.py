# Streamlit app for HS Code prediction

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

# Check for CUDA availability
#if torch.cuda.is_available():
#    device = torch.device("cuda")
#    st.write("CUDA is available. Device:", torch.cuda.get_device_name(0))
#else:
#    st.write("CUDA is not available. Using CPU.")

# CSS for a more visually appealing table
table_css = """
<style>
    table {
        width: 130%;
        border-collapse: collapse;
        margin-left: -40; /* Align table to the left */
        text-align: left; /* Text alignment inside the table */
    }
    th, td {
        padding: 10px;
        border: 1px solid #ddd;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #f1f1f1;
    }
</style>
"""
# Custom CSS for styling the title
title_css = """
<style>
    .title-style {
        font-size: 2.5em;
        color: ##000033;  /* Custom color for the title */
        font-weight: 700;
        text-align: center;
        padding: 3px;
        margin-top: 0;
        background-color: #f0f4f8;  /* Light background behind title */
        border-radius: 8px;
        box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
    }
</style>
"""

# Custom CSS for styling the button
button_css = """
<style>
    .stButton>button {
        color: ##666633;
        background-color: #CCFFCC;  /* Custom button color */
        font-size: 1.2em;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);  /* Subtle shadow */
        transition: background-color 0.3s ease, transform 0.2s ease;  /* Smooth transition */
    }
    .stButton>button:hover {
        background-color: #CCCC66;  /* Darker shade on hover */
        transform: scale(1.05);  /* Slightly larger on hover */
    }
</style>
"""

# Define CSS custom styles
st.markdown(
    """
    <style>
    .text-style {
        color: #A9A9A9; /* Gris claro */
        font-size: 16px;
        font-style: italic; /* Cursiva */
        font-weight: normal;
        text-align: justify;
        margin-top: 10px;
    }
        .highlighted-selection {
        background-color: #e0ffe0; /* Light green background */
        color: #155724; /* Dark green text for readability */
        font-size: 24px;
        font-weight: bold;
        padding: 8px;
        border: 2px solid #28a745; /* Green border */
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 128, 0, 0.2); /* Subtle shadow for effect */
        text-align: center;
        margin-top: 6px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

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

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
model_kwargs = {'device': 'cpu' , 'trust_remote_code': True}
#model_kwargs = {'device': device , 'trust_remote_code': True}
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


# Add CSSs to the page
st.markdown(table_css, unsafe_allow_html=True)
st.markdown(title_css, unsafe_allow_html=True)
st.markdown(button_css, unsafe_allow_html=True)

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
            #Perform Chain Of Thoughts
            # Extract keywords
            keywords = extract_keywords_from_query(query)

            # Translate if necessary
            translated_query = translate_query(keywords)

            # Extract translated keywords
            translated_keywords = extract_keywords_from_query_in_english(translated_query)

            # Retrieve HS code
            result = llm_chain({"question": translated_keywords, "summaries": context})

            # Optional
            #st.write(" On the backend, I'm looking for....")            
            st.markdown(
            f"""
            <div class="text-style">
                {translated_keywords}
            </div>
            """, 
            unsafe_allow_html=True
                    )
            
            st.write("### HS Code apropiado:")
            st.markdown(
                        f"""
                        <div class="highlighted-selection">
                            {result["answer"]}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

            # Display source documents if available
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
