import streamlit as st
import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import PyPDF2

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Detector de Plagio con IA", page_icon="\U0001F50D", layout="wide")

st.title("Detector de Plagio y Similitud Textual con IA")
st.write("Compara dos textos o documentos y detecta el nivel de similitud semantica usando embeddings de IA.")

with st.sidebar:
    st.header("Configuracion")
    api_key_input = st.text_input("OpenAI API Key (opcional si ya esta en .env)", type="password")
    if api_key_input:
        openai.api_key = api_key_input
    umbral = st.slider("Umbral de alerta de plagio (%)", 50, 100, 80)


def extraer_texto_pdf(archivo):
    lector = PyPDF2.PdfReader(archivo)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() or ""
    return texto


def obtener_embedding(texto):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding


col1, col2 = st.columns(2)

with col1:
    st.subheader("Documento / Texto A")
    archivo_a = st.file_uploader("Sube un PDF (opcional)", type=["pdf"], key="a")
    texto_a = st.text_area("O pega el texto A aqui", height=250)
    if archivo_a:
        texto_a = extraer_texto_pdf(archivo_a)

with col2:
    st.subheader("Documento / Texto B")
    archivo_b = st.file_uploader("Sube un PDF (opcional)", type=["pdf"], key="b")
    texto_b = st.text_area("O pega el texto B aqui", height=250)
    if archivo_b:
        texto_b = extraer_texto_pdf(archivo_b)

if st.button("Analizar similitud"):
    if not texto_a or not texto_b:
        st.warning("Por favor proporciona ambos textos o documentos para comparar.")
    else:
        with st.spinner("Analizando similitud semantica..."):
            try:
                emb_a = obtener_embedding(texto_a)
                emb_b = obtener_embedding(texto_b)

                similitud = cosine_similarity([emb_a], [emb_b])[0][0]
                porcentaje = round(similitud * 100, 2)

                st.subheader("Resultado del analisis")
                st.metric("Similitud semantica", f"{porcentaje}%")

                if porcentaje >= umbral:
                    st.error(f"Alerta: el nivel de similitud ({porcentaje}%) supera el umbral definido ({umbral}%). Posible plagio.")
                else:
                    st.success(f"El nivel de similitud ({porcentaje}%) esta por debajo del umbral definido ({umbral}%).")

                st.progress(min(int(porcentaje), 100))

            except Exception as e:
                st.error(f"Error al analizar los textos: {e}")
