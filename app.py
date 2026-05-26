import streamlit as st
import requests
import os
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import uuid
import time
from datetime import datetime

load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = "https://cefis.com.br"
client = Anthropic(api_key=ANTHROPIC_KEY)

@st.cache_resource
def load_rag():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("cefis")
    return model, collection

def carregar_planos(user_id):
    path = f"./dados_{user_id}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"user_id": user_id, "planos": []}

def salvar_plano(user_id, onboarding, plano):
    dados = carregar_planos(user_id)
    novo = {
        "id": str(uuid.uuid4()),
        "titulo": onboarding["objetivo"][:60],
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "onboarding": onboarding,
        "plano": plano
    }
    dados["planos"].append(novo)
    with open(f"./dados_{user_id}.json", "w") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return novo["id"]

def deletar_plano(user_id, plano_id):
    dados = carregar_planos(user_id)
    dados["planos"] = [p for p in dados["planos"] if p["id"] != plano_id]
    with open(f"./dados_{user_id}.json", "w") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def login_cefis(email, senha):
    r = requests.post(f"{BASE_URL}/api/v1/login", json={"email": email, "pass": senha})
    if r.status_code == 200:
        data = r.json()["data"]
        return data["key"], data["user"]
    return None, None

def buscar_conteudo(pergunta, model, collection, n=8):
    embedding = model.encode(pergunta).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    contexto = ""
    for chunk, meta in zip(chunks, metas):
        contexto += f"\n[ID: {meta['curso_id']} | Curso: {meta['curso_titulo']} | Aula: {meta['aula_titulo']}]\n{chunk}\n"
    return contexto

def buscar_cursos(search="", page=1, count=24):
    headers = {"Accept": "application/json"}
    params = {"count": count, "page": page}
    if search:
        params["search"] = search
    r = requests.get("https://api-v3.cefis.com.br/courses", headers=headers, params=params)
    if r.status_code == 200:
        return r.json()
    return None

def renderizar_catalogo(key_prefix=""):
    col_busca, col_qtd = st.columns([4, 1])
    with col_busca:
        busca = st.text_input("🔍 Buscar curso", placeholder="Ex: direito tributário...", key=f"busca_{key_prefix}")
    with col_qtd:
        qtd = st.selectbox("Por página", [12, 24, 48], index=1, key=f"qtd_{key_prefix}")

    pagina_key = f"pagina_{key_prefix}"
    busca_ant_key = f"busca_ant_{key_prefix}"

    if pagina_key not in st.session_state:
        st.session_state[pagina_key] = 1
    if busca_ant_key not in st.session_state:
        st.session_state[busca_ant_key] = ""
    if busca != st.session_state[busca_ant_key]:
        st.session_state[pagina_key] = 1
        st.session_state[busca_ant_key] = busca

    with st.spinner("Carregando cursos..."):
        resultado = buscar_cursos(search=busca, page=st.session_state[pagina_key], count=qtd)

    if resultado:
        cursos = resultado.get("data", [])
        paginacao = resultado.get("pagination", {})
        total_paginas = paginacao.get("lastPage", 20)
        total_itens = paginacao.get("totalItems", 476)
        pagina_atual = st.session_state[pagina_key]
        st.caption(f"Total: {total_itens} cursos · Página {pagina_atual} de {total_paginas}")

        cols = st.columns(3)
        for i, curso in enumerate(cursos):
            with cols[i % 3]:
                if curso.get("banner"):
                    st.image(curso["banner"], use_container_width=True)
                st.markdown(f"**{curso['title']}**")
                st.caption(f"⭐ {curso.get('averageRating', 0):.1f} · {curso.get('lessonCount', 0)} aulas")
                st.link_button("▶️ Acessar", f"https://cefis.com.br/portal/cursos/{curso['id']}", use_container_width=True, key=f"{key_prefix}_{curso['id']}")

        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if pagina_atual > 1:
                if st.button("← Anterior", key=f"ant_{key_prefix}", use_container_width=True):
                    st.session_state[pagina_key] -= 1
                    st.rerun()
        with c2:
            st.markdown(f"<p style='text-align:center'>Página {pagina_atual} de {total_paginas}</p>", unsafe_allow_html=True)
        with c3:
            if pagina_atual < total_paginas:
                if st.button("Próxima →", key=f"prox_{key_prefix}", use_container_width=True):
                    st.session_state[pagina_key] += 1
                    st.rerun()
    else:
        st.error("Não foi possível carregar os cursos.")

def gerar_plano(perfil, onboarding, model, collection):
    tema = onboarding["objetivo"]
    contexto = buscar_conteudo(tema, model, collection, n=12)

    estilo_map = {
        "Visual": "Prefere diagramas, mapas mentais e resumos visuais.",
        "Auditivo": "Aprende melhor ouvindo — prefere podcasts e explicações em áudio.",
        "Lendo/Escrevendo": "Aprende melhor lendo textos e fazendo anotações.",
        "Prático": "Aprende fazendo — prefere exercícios e casos práticos."
    }
    estilo_desc = estilo_map.get(onboarding["estilo"], "")

    prompt = f"""Você é um tutor de aprendizado personalizado da CEFIS.
Monte um plano de estudos completo e personalizado.

PERFIL DO ALUNO:
- Nome: {perfil.get('name')}
- Ocupação: {onboarding['ocupacao']}
- Experiência na área: {onboarding['experiencia']}
- Nível de conhecimento: {onboarding['nivel']}
- Estilo de aprendizagem: {onboarding['estilo']} — {estilo_desc}

OBJETIVO: {onboarding['objetivo']}
TEMPO DISPONÍVEL: {onboarding['tempo']}

CONTEÚDO REAL DISPONÍVEL NA CEFIS (use os IDs e títulos exatos):
{contexto}

Retorne APENAS um JSON válido, sem texto antes ou depois, neste formato exato:

{{
  "diagnostico": {{
    "ja_sabe": "texto sobre o que o aluno já sabe",
    "gaps_criticos": ["gap 1", "gap 2"],
    "gaps_complementares": ["gap 1", "gap 2"]
  }},
  "fases": [
    {{
      "numero": 1,
      "titulo": "título da fase",
      "objetivo": "objetivo da fase",
      "duracao": "X semanas",
      "modulos": [
        {{
          "titulo": "título do módulo",
          "curso_id": "ID numérico do curso da CEFIS ou null se for material da IA",
          "curso_titulo": "título exato do curso da CEFIS ou null",
          "tipo": "cefis ou ia",
          "conteudo": "o que estudar neste módulo",
          "atividade": "atividade de fixação adaptada ao estilo {onboarding['estilo']}"
        }}
      ]
    }}
  ],
  "cronograma": "resumo do cronograma semana a semana",
  "resultado_esperado": "o que o aluno será capaz de fazer ao final"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        messages=[{"role": "user", "content": prompt}]
    )

    texto = response.content[0].text.strip()
    texto = texto.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        try:
            ultimo = max(texto.rfind("}"), texto.rfind("]"))
            texto_cortado = texto[:ultimo+1]
            abertas = texto_cortado.count("{") - texto_cortado.count("}")
            texto_cortado += "}" * abertas
            return json.loads(texto_cortado)
        except:
            return {
                "diagnostico": {
                    "ja_sabe": "Perfil analisado com sucesso.",
                    "gaps_criticos": ["Ver plano abaixo"],
                    "gaps_complementares": []
                },
                "fases": [{
                    "numero": 1,
                    "titulo": "Plano de Estudos",
                    "objetivo": onboarding["objetivo"],
                    "duracao": "A definir",
                    "modulos": [{
                        "titulo": onboarding["objetivo"],
                        "curso_id": None,
                        "curso_titulo": None,
                        "tipo": "ia",
                        "conteudo": "Clique em 'Gerar novo plano' para tentar novamente.",
                        "atividade": ""
                    }]
                }],
                "cronograma": None,
                "resultado_esperado": None
            }

def renderizar_plano(plano, nome):
    st.markdown(f"## 📚 Plano de Estudos — {nome}")

    st.markdown("### 🔍 Diagnóstico de Lacunas")
    st.info(plano["diagnostico"]["ja_sabe"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Gaps Críticos**")
        for g in plano["diagnostico"]["gaps_criticos"]:
            st.markdown(f"- {g}")
    with col2:
        st.markdown("**🟡 Gaps Complementares**")
        for g in plano["diagnostico"]["gaps_complementares"]:
            st.markdown(f"- {g}")

    st.divider()
    st.markdown("### 🗺️ Plano de Estudos")

    for fase in plano["fases"]:
        with st.expander(f"📌 Fase {fase['numero']} — {fase['titulo']} ({fase['duracao']})", expanded=fase['numero'] == 1):
            st.caption(f"🎯 {fase['objetivo']}")
            for mod in fase["modulos"]:
                st.markdown(f"#### 📖 {mod['titulo']}")
                curso_id = mod.get("curso_id")
                try:
                    curso_id_valido = int(str(curso_id)) if curso_id else None
                except:
                    curso_id_valido = None

                if mod["tipo"] == "cefis" and curso_id_valido:
                    url = f"https://cefis.com.br/portal/cursos/{curso_id_valido}"
                    st.markdown(f"🎓 **Curso CEFIS:** [{mod['curso_titulo']}]({url})")
                    st.link_button("▶️ Acessar curso na CEFIS", url)
                else:
                    st.markdown("🤖 **Material gerado pela IA**")
                st.markdown(f"📝 **O que estudar:** {mod['conteudo']}")
                if mod.get("atividade"):
                    st.markdown(f"✏️ **Atividade:** {mod['atividade']}")
                st.divider()

    if plano.get("cronograma"):
        st.markdown("### ⏱️ Cronograma")
        st.markdown(plano["cronograma"])
    if plano.get("resultado_esperado"):
        st.markdown("### 🏆 Resultado Esperado")
        st.success(plano["resultado_esperado"])

def gerar_resposta(mensagens, contexto, perfil, onboarding):
    nivel = onboarding.get("nivel", "intermediário") if onboarding else "intermediário"
    estilo = onboarding.get("estilo", "Lendo/Escrevendo") if onboarding else "Lendo/Escrevendo"

    system = f"""Você é um tutor de aprendizado personalizado da CEFIS.

PERFIL DO ALUNO:
- Nome: {perfil.get('name', 'Aluno')}
- Nível: {nivel}
- Estilo de aprendizagem: {estilo}
- Objetivo: {onboarding.get('objetivo', '') if onboarding else ''}

INSTRUÇÕES:
- Adapte a linguagem ao nível {nivel}
- Se o estilo for Visual: use tabelas, listas e estrutura visual
- Se o estilo for Auditivo: use linguagem conversacional e analogias
- Se o estilo for Prático: foque em exemplos e casos reais
- Sempre fundamente com o conteúdo real da CEFIS abaixo
- Seja direto e didático

CONTEÚDO REAL DA PLATAFORMA CEFIS:
{contexto}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=mensagens
    )
    return response.content[0].text

def gerar_quiz(tema, nivel, model, collection):
    contexto = buscar_conteudo(tema, model, collection, n=6)
    prompt = f"""Com base no conteúdo real da CEFIS abaixo, crie um questionário de 5 perguntas sobre "{tema}".

Nível do aluno: {nivel}

Conteúdo da CEFIS:
{contexto}

Formato:
- 5 questões de múltipla escolha (A, B, C, D)
- Gabarito ao final
- Explicação breve de cada resposta correta
- Baseie as questões no conteúdo real das aulas"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def gerar_resumo(tema, nivel, estilo, model, collection):
    contexto = buscar_conteudo(tema, model, collection, n=8)
    prompt = f"""Com base no conteúdo real das aulas da CEFIS, crie uma apostila resumida sobre "{tema}".

Nível: {nivel}
Estilo de aprendizagem: {estilo}

Conteúdo das aulas:
{contexto}

Se o estilo for Visual: use tabelas comparativas, listas estruturadas e hierarquia clara.
Se o estilo for Auditivo: use linguagem conversacional, como se estivesse explicando em voz alta.
Se o estilo for Prático: foque em exemplos reais, casos e aplicações.
Se o estilo for Lendo/Escrevendo: use texto corrido bem estruturado com subtítulos.

Inclua: conceitos principais, exemplos práticos e pontos de atenção."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# CONFIG
st.set_page_config(page_title="Tutor CEFIS", page_icon="🎓", layout="wide")

# SESSION STATE
for key, default in [
    ("logado", False), ("api_key", None), ("usuario", {}),
    ("mensagens", []), ("plano", None), ("onboarding", None),
    ("onboarding_completo", False), ("plano_ativo_id", None),
    ("tela", "catalogo")
]:
    if key not in st.session_state:
        st.session_state[key] = default

# LOGIN
if not st.session_state.logado:
    st.title("🎓 Tutor CEFIS")
    st.subheader("Entre com sua conta CEFIS para começar")

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

# CATÁLOGO PRÉ-ONBOARDING
elif st.session_state.tela == "catalogo":
    user = st.session_state.usuario
    st.title(f"Olá, {user.get('first_name', 'Aluno')}! 👋")

    dados = carregar_planos(user["id"])
    planos = dados["planos"]

    if planos:
        st.markdown("### 📋 Seus planos de estudo")
        for p in planos:
            col1, col2, col3 = st.columns([5, 2, 1])
            with col1:
                st.markdown(f"**🎯 {p['titulo']}**")
                st.caption(f"Criado em {p['created_at']} · Nível {p['onboarding']['nivel']} · {p['onboarding']['estilo']}")
            with col2:
                if st.button("▶️ Abrir", key=f"abrir_{p['id']}"):
                    st.session_state.onboarding = p["onboarding"]
                    st.session_state.plano = p["plano"]
                    st.session_state.plano_ativo_id = p["id"]
                    st.session_state.onboarding_completo = True
                    st.session_state.tela = "app"
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_{p['id']}"):
                    deletar_plano(user["id"], p["id"])
                    st.rerun()
        st.divider()

    if st.button("➕ Criar novo plano de estudos", use_container_width=True):
        st.session_state.tela = "onboarding"
        st.session_state.onboarding = None
        st.session_state.plano = None
        st.rerun()

    st.divider()
    st.markdown("### 📚 Cursos disponíveis na CEFIS")
    renderizar_catalogo(key_prefix="pre")

# ONBOARDING
elif st.session_state.tela == "onboarding":
    user = st.session_state.usuario
    st.title("🎯 Novo Plano de Estudos")
    st.subheader("Me conta sobre você e seu objetivo")

    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        with col1:
            ocupacao = st.text_input("Qual é a sua ocupação atual?", placeholder="Ex: Advogado, Contador...")
            experiencia = st.selectbox("Experiência na área?", [
                "Nenhuma — estou começando do zero",
                "Pouca — já ouvi falar mas não domino",
                "Moderada — conheço o básico",
                "Avançada — já trabalho ou estudei a fundo"
            ])
            nivel = st.selectbox("Como você se classifica?", ["Iniciante", "Intermediário", "Avançado"])
        with col2:
            objetivo = st.text_area("Qual é o seu objetivo?", placeholder="Ex: Passar na OAB, concurso fiscal...")
            tempo = st.selectbox("Tempo disponível para estudar?", [
                "10 minutos agora", "30 minutos por dia",
                "1 hora por dia", "2 horas por dia", "Mais de 2 horas por dia"
            ])
            estilo = st.selectbox("Como você aprende melhor?", ["Visual", "Auditivo", "Lendo/Escrevendo", "Prático"])

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            voltar = st.form_submit_button("← Voltar", use_container_width=True)
        with col_btn2:
            comecar = st.form_submit_button("🚀 Gerar plano personalizado", use_container_width=True)

    if voltar:
        st.session_state.tela = "catalogo"
        st.rerun()

    if comecar and objetivo and ocupacao:
        st.session_state.onboarding = {
            "ocupacao": ocupacao, "experiencia": experiencia,
            "nivel": nivel, "objetivo": objetivo,
            "tempo": tempo, "estilo": estilo
        }
        progress_bar = st.progress(0, text="🔍 Analisando seu perfil...")
        time.sleep(0.8)
        progress_bar.progress(15, text="📚 Buscando cursos relevantes na CEFIS...")
        time.sleep(0.8)
        progress_bar.progress(35, text="🧠 Identificando lacunas de conhecimento...")
        time.sleep(0.8)
        progress_bar.progress(55, text="🗺️ Montando sequência de estudos...")
        time.sleep(0.8)
        progress_bar.progress(75, text="✍️ Gerando atividades personalizadas...")
        model, collection = load_rag()
        plano = gerar_plano(user, st.session_state.onboarding, model, collection)
        progress_bar.progress(95, text="💾 Salvando seu plano...")
        time.sleep(0.5)
        progress_bar.progress(100, text="✅ Plano pronto!")
        time.sleep(0.5)
        st.session_state.plano = plano
        plano_id = salvar_plano(user["id"], st.session_state.onboarding, plano)
        st.session_state.plano_ativo_id = plano_id
        st.session_state.onboarding_completo = True
        st.session_state.tela = "app"
        st.rerun()
    elif comecar:
        st.warning("Preencha pelo menos o objetivo e a ocupação.")

# APP PRINCIPAL
elif st.session_state.tela == "app":
    model, collection = load_rag()
    user = st.session_state.usuario
    onboarding = st.session_state.onboarding

    st.sidebar.title(f"{user.get('first_name', 'Aluno')} 👋")
    st.sidebar.caption(f"🎯 {onboarding['objetivo'][:50]}...")
    st.sidebar.caption(f"📚 Nível: {onboarding['nivel']}")
    st.sidebar.caption(f"🧠 Estilo: {onboarding['estilo']}")
    st.sidebar.divider()

    pagina = st.sidebar.radio("Menu", [
        "📋 Plano de Estudos",
        "💬 Tirar Dúvidas",
        "📝 Questionário",
        "📄 Apostila Resumida",
        "📚 Catálogo de Cursos"
    ])

    if st.sidebar.button("← Meus planos"):
        st.session_state.tela = "catalogo"
        st.rerun()

    if st.sidebar.button("➕ Novo plano"):
        st.session_state.tela = "onboarding"
        st.session_state.onboarding = None
        st.session_state.plano = None
        st.rerun()

    if st.sidebar.button("Sair"):
        for k in ["logado", "api_key", "usuario", "mensagens", "plano", "onboarding", "onboarding_completo", "plano_ativo_id"]:
            st.session_state[k] = None
        st.session_state.logado = False
        st.session_state.tela = "catalogo"
        st.rerun()

    if pagina == "📋 Plano de Estudos":
        st.title("📋 Seu Plano de Estudos Personalizado")
        if not st.session_state.plano:
            with st.spinner("Montando seu plano..."):
                st.session_state.plano = gerar_plano(user, onboarding, model, collection)
        renderizar_plano(st.session_state.plano, user.get('name', 'Aluno'))
        if st.button("🔄 Gerar novo plano"):
            st.session_state.plano = None
            st.rerun()

    elif pagina == "💬 Tirar Dúvidas":
        st.title("💬 Tire suas Dúvidas")
        st.caption("Respostas baseadas no conteúdo real das aulas da CEFIS")
        for msg in st.session_state.mensagens:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        pergunta = st.chat_input("Pergunte sobre qualquer conteúdo da CEFIS...")
        if pergunta:
            st.session_state.mensagens.append({"role": "user", "content": pergunta})
            with st.chat_message("user"):
                st.write(pergunta)
            with st.chat_message("assistant"):
                with st.spinner("Buscando nas aulas da CEFIS..."):
                    contexto = buscar_conteudo(pergunta, model, collection)
                    resposta = gerar_resposta(st.session_state.mensagens, contexto, user, onboarding)
                st.write(resposta)
            st.session_state.mensagens.append({"role": "assistant", "content": resposta})

    elif pagina == "📝 Questionário":
        st.title("📝 Questionário Personalizado")
        st.caption("Gerado com base no conteúdo real das aulas da CEFIS")
        with st.form("quiz"):
            tema_quiz = st.text_input("Sobre qual tema quer ser testado?", value=onboarding["objetivo"])
            gerar_q = st.form_submit_button("Gerar Questionário")
        if gerar_q:
            with st.spinner("Gerando questionário..."):
                quiz = gerar_quiz(tema_quiz, onboarding["nivel"], model, collection)
            st.markdown(quiz)

    elif pagina == "📄 Apostila Resumida":
        st.title("📄 Apostila Resumida")
        st.caption("Gerada com base no conteúdo real das aulas da CEFIS")
        with st.form("apostila"):
            tema_ap = st.text_input("Sobre qual tema quer a apostila?", value=onboarding["objetivo"])
            gerar_a = st.form_submit_button("Gerar Apostila")
        if gerar_a:
            with st.spinner("Gerando apostila..."):
                resumo = gerar_resumo(tema_ap, onboarding["nivel"], onboarding["estilo"], model, collection)
            st.markdown(resumo)

    elif pagina == "📚 Catálogo de Cursos":
        st.title("📚 Catálogo de Cursos CEFIS")
        renderizar_catalogo(key_prefix="app")