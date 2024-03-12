from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
import pickle
import streamlit as st
import os
import time
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("https://github.com/harshitv804/RAG-without-Vector-DB/assets/100853494/cebecf35-bdd6-459c-8a1e-bad7c9a8037b")

st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffffff;
    color: #000000
}
div.stButton > button:active {
    background-color: #ff6262;
}
   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 


with open('cosmos_data.dump', 'rb') as f:
    doc = pickle.load(f)

flash_rerank = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2",top_n=8)
reordering = LongContextReorder()

tfid_retriever = TFIDFRetriever.from_documents(doc,k=50)

pipeline = DocumentCompressorPipeline(transformers=[reordering,flash_rerank])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=tfid_retriever
)

prompt_template = """<s>[INST]This is a chat template and you are the guide to the universe, your primary objective is to provide accurate and concise information related to universe based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the universe.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

llm = ChatGroq(temperature=0.7,max_tokens=1024, groq_api_key= os.environ['GROQ_API'], model_name="mixtral-8x7b-32768")

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=compression_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})
