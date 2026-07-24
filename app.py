import os
import time
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_vector_store():
    # Fetch configurations
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable is missing.")

    # Initialize embeddings
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=cohere_api_key
    )

    index_path = "./faiss_index"

    # Strictly load the pre-built index. Fail fast if it does not exist.
    if not os.path.exists(index_path):
        raise RuntimeError(
            f"Critical Error: FAISS index not found at {index_path}. "
            "The vector database must be built and deployed alongside the application."
        )

    # Load the static index into memory
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


@st.cache_resource
def setup_rag_chain(_vectorstore):
    cohere_api_key = os.getenv("COHERE_API_KEY")
    model_name = os.getenv("COHERE_MODEL_NAME")

    if not model_name:
        raise ValueError("COHERE_MODEL_NAME is missing in .env")

    # Initialize LLM
    llm = ChatCohere(
        model=model_name,
        temperature=0.1,
        cohere_api_key=cohere_api_key
    )

    # Configure the retriever with a calibrated relevance threshold
    retriever = _vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.2, "k": 4}
    )

    # 1. Chain to reformulate the question with history context
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Sub-chain that outputs the rewritten string
    history_aware_retriever_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Main Q&A Prompt
    system_prompt = (
        "Eres un agente de soporte técnico para BimBam Buy. "
        "Tu única tarea es responder consultas basándote en la información proporcionada.\n\n"
        "<CONTEXTO_DISPONIBLE>\n"
        "{context}\n"
        "</CONTEXTO_DISPONIBLE>\n\n"
        "DIRECTRICES DE OPERACIÓN:\n"
        "1. Tono: Dirígete al usuario en segunda persona formal ('usted').\n"
        "2. Adaptación: Explica las políticas de forma natural. Omite referencias estructurales como 'sección 1' o 'artículo 3'.\n"
        "3. Precisión Terminológica: Utiliza estrictamente el vocabulario presente en el contexto. Tienes prohibido asumir sinónimos comerciales o prometer exenciones.\n"
        "4. Inmersión de Rol: NUNCA uses frases como 'según el contexto proporcionado', 'en mis documentos', o 'el texto dice'. Habla directamente en nombre de las políticas de BimBam Buy.\n"
        "5. Acción por defecto (Contexto Vacío): Si el bloque <CONTEXTO_DISPONIBLE> está vacío, carece de información relevante, o el tema no se menciona en absoluto, es MANDATORIO que generes exactamente esta cadena de texto para evitar errores de sistema: "
        "'Lo siento, no tengo acceso a esa información específica en mis manuales actuales.'\n"
        "Bajo ninguna circunstancia debes dejar la respuesta en blanco ni intentar deducir información externa."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Assemble complete RAG chain using pure LCEL
    # The input passes through the reformulation chain, then the retriever, then formatting
    rag_chain = (
            RunnablePassthrough.assign(
                context=history_aware_retriever_chain | retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain



# --- User Interface & State Management ---
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("robot.png", width=170)

st.markdown("""
<div style="
background: linear-gradient(90deg,#4F46E5,#2563EB);
padding:20px;
border-radius:15px;
color:white;
text-align:center;
margin-bottom:20px;
">
<h2>BimBam Buy Support Agent</h2>
<p>Pregúntame sobre productos, políticas, envíos, devoluciones y mucho más.</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history early
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:

    st.info("""
👋 ¡Bienvenido!

Soy el asistente virtual de **BimBam Buy**.

Puedo ayudarte con:

- 📦 Productos
- 🚚 Envíos
- 💳 Pagos
- 🔄 Devoluciones
- ❓ Preguntas frecuentes

Escribe tu consulta para comenzar.
""")

# Render existing messages IMMEDIATELY to prevent visual clearing
for message in st.session_state.messages:

    avatar = "🤖" if message["role"] == "assistant" else "🧑"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Load cached resources AFTER rendering the UI
try:
    v_store = initialize_vector_store()
    rag_chain = setup_rag_chain(v_store)
except Exception as e:
    st.error(f"System initialization error: {e}")
    st.stop()

# Capture user input
if user_query := st.chat_input("¿En qué puedo ayudarte hoy?"):

    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Analizando la información..."):
            try:
                # Extract history and apply pruning
                langchain_history = []
                for msg in st.session_state.messages[-7:-1]:
                    if msg["role"] == "user":
                        langchain_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_history.append(AIMessage(content=msg["content"]))

                # Execute pipeline
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": langchain_history
                })

                st.markdown(response)

                # Persist assistant response only on successful API execution
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # Rollback state to maintain alternating role integrity for Cohere API
                st.session_state.messages.pop()
                st.error(f"System error: {e}. State rolled back to prevent history corruption.")