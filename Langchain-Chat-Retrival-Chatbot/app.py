import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders.csv_loader import CSVLoader
import requests
from PIL import Image
import time

# Set Streamlit page configuration
st.set_page_config(
   page_title="FoodGPT - Nagpur Based Food Recommendation System.",
   page_icon="🍊",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Load data and set up language chain components
loader = CSVLoader(file_path='data.csv')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

vector_store = FAISS.from_documents(text_chunks, embeddings)

llm = CTransformers(model="model.bin", model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

# Sidebar for user input and query history
st.sidebar.title("FoodGPT!🍊")
st.sidebar.info("FoodGPT : A Nagpur Based Food Recommendation Chat! Recommends you the best locally recognized brands for your cravings! As this system is backed with LLMA-2 on hand-picked data.")
github_link = "[GitHub]()"
st.sidebar.info("To contribute and Sponsor - " + github_link)

# Add a queue to store past queries in the sidebar
if 'query_queue' not in st.session_state:
    st.session_state['query_queue'] = []

# Display past queries in the sidebar
st.sidebar.subheader("Past Queries:")
for i, past_query in enumerate(st.session_state['query_queue']):
    st.sidebar.write(f"{i + 1}. {past_query}")

st.title("FoodGPT: A Nagpur based Food Recommendation Bot! 🍊")

# Initialize session state if not present
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! I'm FoodGPT, Ask me anything about Nagpur's Food."]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hello!"]

reply_container = st.container()
container = st.container()

with container:
    with st.form(key='user_form', clear_on_submit=True):
        user_input = st.text_input("Question:", placeholder="Ask anything about Nagpur's Food Joints or cravings", key='input_user')
        image_upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        submit_button_user = st.form_submit_button(label='Send')

    try:
        if submit_button_user and user_input:
            # Update the queue with the new query
            st.session_state['query_queue'].append(user_input)

            # Retrieve and display the bot's response
            output = chain({"question": user_input, "chat_history": st.session_state['history']})["answer"]
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            user_message = st.session_state["past"][i]
            generated_message = st.session_state["generated"][i]

            # Display user message without any avatar
            st.write(user_message)

            # Display bot-generated message without any avatar
            st.write(generated_message)

import requests

API_URL = "https://api-inference.huggingface.co/models/Prasanna18/indian-food-classification"
HEADERS = {"Authorization": "Bearer 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"}

def query_image_classification(image_bytes, max_retries=3):
    for retry in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
            response.raise_for_status()  # Raise an error for non-2xx HTTP responses
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the API request: {str(e)}")
        except ValueError as e:
            st.error(f"An error occurred while processing the API response: {str(e)}")
        if retry < max_retries - 1:
            st.warning(f"Retrying request (attempt {retry + 1}/{max_retries})...")
            time.sleep(1)  # Wait for a moment before retrying
    st.error("No classification result received after multiple retries.")
    return None

if image_upload:
    image_bytes = image_upload.read()

    classification_result = query_image_classification(image_bytes)

    if classification_result:
        st.image(image_upload, caption="Uploaded Image", use_column_width=True)

        if isinstance(classification_result, list) and classification_result:
            # Ensure that classification_result is a list of results and not empty
            best_label = max(classification_result, key=lambda x: x.get('score', 0))

            if 'label' in best_label:
                st.header("Image Classification Result:")
                st.write(f"Classified as: {best_label['label']}")
            else:
                st.error("Invalid classification result format. Missing 'label' key.")
        else:
            st.error("Invalid classification result format or empty result list.")
    else:
        st.error("No classification result received.")
