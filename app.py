import streamlit as st
import requests
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Configuração
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = "https://cefis.com.br"
API_V3 = "https://api-v3.cefis.com.br"

client = Anthropic(api_key=ANTHROPIC_KEY)

@st.cache_resource
def load_rag():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("cefis")
    return model, collection

def login_cefis(email, senha):
    r = requests.post(f"{BASE_URL}/api/v1/login", json={"email": email, "pass": senha})
    if r.status_code == 200:
        data = r.json()["data"]
        return data["key"], data["user"]
    return None, None

def buscar_conteudo(pergunta, model, collection, n=5):
    embedding = model.encode(pergunta).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    contexto = ""
    for chunk, meta in zip(chunks, metas):
        contexto += f"\n[{meta['curso_titulo']} — {meta['aula_titulo']}]\n{chunk}\n"
    return contexto

def gerar_resposta(mensagens, contexto, perfil):
    system = f"""Você é um tutor de aprendizado personalizado da CEFIS.

Perfil do aluno:
- Nome: {perfil.get('name', 'Aluno')}
- Nível: {perfil.get('nivel', 1)}
- Ocupação: {perfil.get('occupation', 'Não informado')}
- Áreas: {perfil.get('activities', [])}

Use o conteúdo real da CEFIS abaixo para fundamentar suas respostas.
Seja direto, didático e adaptado ao nível do aluno.

Conteúdo relevante da plataforma:
{contexto}
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=mensagens
    )
    return response.content[0].text

def gerar_plano(perfil, objetivo, tempo_disponivel, model, collection):
    contexto = buscar_conteudo(objetivo, model, collection, n=10)
    prompt = f"""Com base no perfil e objetivo do aluno, crie um plano de estudos personalizado.

Perfil:
- Nome: {perfil.get('name')}
- Ocupação: {perfil.get('occupation', 'Não informado')}
- Nível: {perfil.get('nivel', 1)}

Objetivo: {objetivo}
Tempo disponível: {tempo_disponivel}

Conteúdo disponível na CEFIS:
{contexto}

Monte um plano com:
1. Diagnóstico do que o aluno precisa aprender
2. Sequência de estudos com cursos reais da CEFIS
3. Estimativa de tempo por etapa
4. O que o aluno será capaz de fazer ao final

Seja específico e use os nomes reais dos cursos encontrados."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Interface
st.set_page_config(page_title="Tutor CEFIS", page_icon="🎓", layout="wide")

# Session state
if "logado" not in st.session_state:
    st.session_state.logado = False
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "usuario" not in st.session_state:
    st.session_state.usuario = {}
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "plano" not in st.session_state:
    st.session_state.plano = None

# LOGIN
if not st.session_state.logado:
    st.title("🎓 Tutor CEFIS")
    st.subheader("Entre com sua conta CEFIS")
    
    with st.form("login"):
        email = st.text_input("E-mail")
        senha = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")
    
    if entrar:
        with st.spinner("Autenticando..."):
            key, user = login_cefis(email, senha)
        if key:
            st.session_state.logado = True
            st.session_state.api_key = key
            st.session_state.usuario = user
            st.rerun()
        else:
            st.error("E-mail ou senha incorretos.")

# APP PRINCIPAL
else:
    model, collection = load_rag()
    user = st.session_state.usuario

    st.sidebar.title(f"Olá, {user.get('first_name', 'Aluno')} 👋")
    st.sidebar.write(f"Nível {user.get('nivel', 1)}")
    st.sidebar.divider()
    pagina = st.sidebar.radio("Menu", ["📋 Plano de Estudos", "💬 Tirar Dúvidas"])
    
    if st.sidebar.button("Sair"):
        st.session_state.logado = False
        st.rerun()

    # PLANO DE ESTUDOS
    if pagina == "📋 Plano de Estudos":
        st.title("📋 Seu Plano de Estudos")
        
        with st.form("onboarding"):
            objetivo = st.text_area(
                "Qual é o seu objetivo de aprendizado?",
                placeholder="Ex: Quero me preparar para o exame da OAB em direito tributário"
            )
            tempo = st.selectbox(
                "Quanto tempo você tem disponível para estudar?",
                ["30 minutos por dia", "1 hora por dia", "2 horas por dia", "Mais de 2 horas por dia", "Tenho 10 minutos agora"]
            )
            gerar = st.form_submit_button("Gerar Plano de Estudos")
        
        if gerar and objetivo:
            with st.spinner("Montando seu plano personalizado..."):
                plano = gerar_plano(user, objetivo, tempo, model, collection)
                st.session_state.plano = plano
        
        if st.session_state.plano:
            st.markdown(st.session_state.plano)

    # CHAT
    elif pagina == "💬 Tirar Dúvidas":
        st.title("💬 Tire suas Dúvidas")
        
        for msg in st.session_state.mensagens:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        pergunta = st.chat_input("Pergunte qualquer coisa sobre o conteúdo da CEFIS...")
        
        if pergunta:
            st.session_state.mensagens.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.write(pergunta)
            
            with st.chat_message("assistant"):
                with st.spinner("Buscando no conteúdo da CEFIS..."):
                    contexto = buscar_conteudo(pergunta, model, collection)
                    resposta = gerar_resposta(st.session_state.mensagens, contexto, user)
                st.write(resposta)
            
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})
