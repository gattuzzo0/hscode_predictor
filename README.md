# Streamlit HS Codes Prediction

This project aims to develop a predictive system for HS (Harmonized System) codes using a Streamlit interface. The model leverages natural language processing techniques and embeddings to predict HS codes based on descriptions, particularly for international shipping manifests. The repository includes data extraction, model training, and a Streamlit interface to facilitate user interaction.

# Brief Description of Models used
# Alibaba-NLP/gte-large-en-v1.5: 
This sentence transformer model, developed by Alibaba, is designed to produce high-quality embeddings for English text. It is based on the GTE (General Text Embedding) architecture and trained to generate dense vector 
representations that capture semantic meanings, making it particularly useful for tasks requiring sentence similarity and retrieval. In this project, it is used to encode HS code descriptions, enabling more accurate retrieval and matching for prediction purposes.

# Llama 3.1: 
Llama 3.1 is an advanced large language model (LLM) developed by Meta. Known for its strong performance across a range of natural language processing tasks, Llama 3.1 builds on previous versions with improvements in language understanding and generation. 
While not directly used in this project, it’s noteworthy as a competitive model for tasks involving complex text generation, summarization, and classification. Its capabilities make it a candidate for future model iterations or expansions of this project.


## Project Structure
  
- **data/WebScrap_CSVs/**: Directory for raw data files collected via web scraping.
  - `hts_codes_WebScrapped.csv`: CSV file with web-scraped HS codes.
  - `hts_dictionary_extended.csv`: Extended dictionary for HTS (Harmonized Tariff Schedule) codes. (NOT USED)

- **embedding/**: Directory for notebooks and scripts related to embeddings.
  - `Alibaba_embeddings.ipynb`: Jupyter notebook for generating and testing embeddings for the HS codes, likely using Alibaba or other embedding models.

- **index/Alibaba/**: Directory containing the Chroma index used as a retriever for RAG (Retrieval-Augmented Generation) tasks.
  - `chroma.sqlite3`: SQLite database file storing the Chroma index.
  - Subdirectory for storing specific indexes related to embedding vectors and search functionalities.

- **src/**: Source code directory for the main scripts.
  - `hscode_UI.py`: Streamlit application script for the user interface, where users can input descriptions and receive HS code predictions.

- **.gitignore**: Specifies files and directories to be ignored by Git, including `.venv/` and other potentially sensitive or large files.

- **requirements.txt**: Lists all dependencies needed for the project, which can be installed using `pip install -r requirements.txt`.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Streamlit_HS_Codes
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run src/hscode_UI.py
   ```
   This will start the Streamlit application, where you can input product descriptions to predict the corresponding HS code.

2. **Data and Model Preparation**:
   - The `embedding/Alibaba_embeddings.ipynb` notebook is used to generate embeddings for the HS codes using sentence-transformers or other NLP models. Ensure embeddings are generated and stored in the `index/Alibaba/` directory for the Chroma retriever to work effectively.
   - The Chroma index in `index/Alibaba/chroma.sqlite3` provides efficient retrieval of embeddings for accurate HS code prediction.

## Notes

- This project uses the Chroma index as a retriever, which is critical for enhancing the accuracy of the prediction by efficiently searching through embeddings.
- The HS codes data may need regular updates as international tariffs and regulations change.
  
## Future Improvements

- Improve data quality for better predictions by enriching the descriptions.
- Implement additional language models for embeddings as they become available.
- Consider adding further data sources and expanding the dataset to include more detailed descriptions and examples.

## Rights and Academic Use
This project has been developed as part of an academic assignment for Tec de Monterrey in the course "Integrative Project" under the supervision of Horacio Martínez Alfaro. Its use is limited exclusively to educational purposes, and commercial exploitation or distribution outside the academic sphere is not authorized without the express consent of the authors and the university.

Any reference to or reuse of this work must be properly cited and credited to the original authors. For inquiries or requests for use, please contact the authors through the official channels of the university.

---
