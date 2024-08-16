import os
import nest_asyncio
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.readers.json import JSONReader
from PIL import Image

# Aplicar nest_asyncio
nest_asyncio.apply()

# Configurar la API Key de OpenAI desde el archivo .env
os.environ["OPENAI_API_KEY"] = dotenv_values(".env")["API_KEY"]

# Configurar la interfaz de Streamlit
st.set_page_config(page_title="Chatbot Médico", page_icon=":hospital:")

# Función para cargar datos del archivo JSON
@st.cache_data
def load_data(file_path):
    return pd.read_json(file_path)

# Barra lateral
st.sidebar.title("Información")
st.sidebar.info(
    """
    Este Chatbot Médico utiliza un modelo LLM para predecir y consultar información médica relevante. 
    Simplemente ingrese la información relevante en el campo de texto y seleccione el tipo de pregunta que desea hacer.
    """
)

st.sidebar.title("Configuración")
st.sidebar.markdown("")

# Selección del tipo de pregunta en la barra lateral
st.sidebar.markdown("### Seleccione el tipo de pregunta:")
prompt_type = st.sidebar.radio("", [
    "Predicción de Enfermedades Basadas en Síntomas",
    "Predicción de Enfermedades Basadas en Resultados de Laboratorio",
    "Predicción de Enfermedades Basadas en Antecedentes",
    "Sugerencias de Pruebas",
    "Información sobre Enfermedades y Resultados del Laboratorio Estándares"
]) 
# Autocompletado según la selección de la pregunta
if prompt_type == "Predicción de Enfermedades Basadas en Síntomas":
    default_text = "Paciente ingresa con síntomas de..."
elif prompt_type == "Predicción de Enfermedades Basadas en Resultados de Laboratorio":
    default_text = "Los resultados de laboratorio muestran..."
elif prompt_type == "Predicción de Enfermedades Basadas en Antecedentes":
    default_text = "El paciente tiene antecedentes de..."
elif prompt_type == "Sugerencias de Pruebas":
    default_text = "El paciente presenta los siguientes síntomas y pruebas anteriores..."
elif prompt_type == "Información sobre Enfermedades y Resultados del Laboratorio Estándares":
    default_text = "La enfermedad en cuestión es..."

# Título y descripción en la página principal
st.title("Chatbot Médico :hospital:")
st.write("Introduzca información relevante sobre el paciente y seleccione el tipo de pregunta en la barra lateral.")

# Subir archivo JSON
json_path = os.path.join("data", "historias100.json")

# Cargar los datos del JSON
data = load_data(json_path)

# Inicializar JSONReader
reader = JSONReader(
    levels_back=0,
    collapse_length=None,
    ensure_ascii=False,
    is_jsonl=False,
    clean_json=True
)

# Cargar documentos desde el archivo JSON
documents = reader.load_data(input_file=json_path, extra_info={})

# Definir el modelo LLM
llm = OpenAI(model="gpt-3.5-turbo")
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents=documents)
vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=5)

# Variables de sesión para almacenar la conversación
if "history" not in st.session_state:
    st.session_state.history = []

def display_history():
    if st.session_state.history:
        st.markdown("### Historial de la conversación:")
        for entry in reversed(st.session_state.history):  # Mostrar en orden inverso
            if "user" in entry:
                st.markdown(f"""
                <div style="background-color:#f0f0f5; color:#000; padding:10px; border-radius:10px; margin-bottom:10px; width:80%; margin-left:20%;">
                    <strong>Usuario:</strong> {entry['user']}
                </div>
                """, unsafe_allow_html=True)
            if "bot" in entry:
                st.markdown(f"""
                <div style="background-color:#d1e7dd; color:#000; padding:10px; border-radius:10px; margin-bottom:10px; width:80%;">
                    <strong>Bot:</strong> {entry['bot']}
                </div>
                """, unsafe_allow_html=True)

# Campo de entrada en la parte inferior con autocompletado
user_input = st.text_area("Introduce la información relevante:", value=default_text)

# Botón para enviar la consulta
if st.button("Enviar"):
    if user_input:
        if prompt_type == "Predicción de Enfermedades Basadas en Síntomas":
            prompt = f"A partir de la siguiente descripción médica, enumera las 5 enfermedades más probables que podrían estar asociadas con estos síntomas: {user_input}."
        elif prompt_type == "Predicción de Enfermedades Basadas en Resultados de Laboratorio":
            prompt = f"Los siguientes son los resultados de laboratorio de un paciente: {user_input}. Basándote en estos resultados, ¿qué niveles específicos deberían ser revisados y qué condiciones podrían estar relacionadas con estos resultados?"
        elif prompt_type == "Predicción de Enfermedades Basadas en Antecedentes":
            prompt = f"Considerando la historia médica del paciente y la descripción proporcionada: {user_input}, ¿cuáles son las 5 enfermedades más probables que deberían ser consideradas?"
        elif prompt_type == "Sugerencias de Pruebas":
            prompt = f"A partir de la siguiente información sobre un paciente: {user_input}, ¿cuáles son las 5 recomendaciones de pruebas que se que se deberían realizar para confirmar un diagnóstico?"
        elif prompt_type == "Información sobre Enfermedades y Resultados del Laboratorio Estándares":
            prompt = f"Para la enfermedad {user_input}, proporciona información sobre los niveles de resultados de laboratorio típicos y cómo estos pueden variar según la condición del paciente."

        # Registrar la pregunta en el historial
        st.session_state.history.append({"user": user_input})

        # Obtener respuesta
        response_vector = query_engine.query(prompt)
        response = response_vector.response

        # Registrar la respuesta en el historial
        st.session_state.history.append({"bot": response})

        # Mostrar el historial
        display_history()

# Añadir estilo con CSS
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f5;
        }
        .main .block-container {
            padding: 2rem 1rem;
        }
        .stButton>button {
            color: white;
            background-color: #007bff;
            border: none;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stTextArea textarea {
            font-size: 1rem;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)