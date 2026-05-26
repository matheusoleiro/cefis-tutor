from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import requests
import os
import json
import uuid
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
BASE_URL = "https://cefis.com.br"

# Load RAG once at startup
print("Carregando modelo de embeddings...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("cefis")
print("Modelo carregado.")

# ── Models ──────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class OnboardingRequest(BaseModel):
    user_id: int
    user_name: str
    ocupacao: str
    experiencia: str
    nivel: str
    objetivo: str
    tempo: str
    estilo: str

class ChatRequest(BaseModel):
    pergunta: str
    historico: list
    user_name: str
    nivel: str
    estilo: str
    objetivo: str

class ContentRequest(BaseModel):
    tema: str
    nivel: str
    estilo: str
    tipo: str  # "quiz" ou "apostila"

class DeletePlanRequest(BaseModel):
    user_id: int
    plano_id: str

# ── Helpers ──────────────────────────────────────────────
def buscar_conteudo(pergunta, n=8):
    embedding = model.encode(pergunta).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n)
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    contexto = ""
    for chunk, meta in zip(chunks, metas):
        contexto += f"\n[ID: {meta['curso_id']} | Curso: {meta['curso_titulo']} | Aula: {meta['aula_titulo']}]\n{chunk}\n"
    return contexto

def carregar_planos(user_id):
    path = f"./dados_{user_id}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"user_id": user_id, "planos": []}

def salvar_plano_db(user_id, onboarding, plano):
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
    return novo

# ── Routes ──────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/api/login")
def login(req: LoginRequest):
    r = requests.post(f"{BASE_URL}/api/v1/login", json={"email": req.email, "pass": req.password})
    if r.status_code == 200:
        data = r.json()["data"]
        return {"ok": True, "key": data["key"], "user": data["user"]}
    raise HTTPException(status_code=401, detail="Credenciais inválidas")

@app.get("/api/cursos")
def cursos(search: str = "", page: int = 1, count: int = 24):
    params = {"count": count, "page": page}
    if search:
        params["search"] = search
    r = requests.get("https://api-v3.cefis.com.br/courses", params=params)
    if r.status_code == 200:
        return r.json()
    raise HTTPException(status_code=500, detail="Erro ao buscar cursos")

@app.get("/api/planos/{user_id}")
def get_planos(user_id: int):
    return carregar_planos(user_id)

@app.post("/api/planos/deletar")
def deletar_plano(req: DeletePlanRequest):
    dados = carregar_planos(req.user_id)
    dados["planos"] = [p for p in dados["planos"] if p["id"] != req.plano_id]
    with open(f"./dados_{req.user_id}.json", "w") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return {"ok": True}

@app.post("/api/gerar-plano")
def gerar_plano(req: OnboardingRequest):
    contexto = buscar_conteudo(req.objetivo, n=12)

    estilo_map = {
        "Visual": "Prefere diagramas, mapas mentais e resumos visuais.",
        "Auditivo": "Aprende melhor ouvindo — prefere podcasts e explicações em áudio.",
        "Lendo/Escrevendo": "Aprende melhor lendo textos e fazendo anotações.",
        "Prático": "Aprende fazendo — prefere exercícios e casos práticos."
    }
    estilo_desc = estilo_map.get(req.estilo, "")

    prompt = f"""Você é um tutor de aprendizado personalizado da CEFIS.
Monte um plano de estudos completo e personalizado.

PERFIL DO ALUNO:
- Nome: {req.user_name}
- Ocupação: {req.ocupacao}
- Experiência na área: {req.experiencia}
- Nível de conhecimento: {req.nivel}
- Estilo de aprendizagem: {req.estilo} — {estilo_desc}

OBJETIVO: {req.objetivo}
TEMPO DISPONÍVEL: {req.tempo}

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
          "atividade": "atividade de fixação adaptada ao estilo {req.estilo}"
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

    texto = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()

    try:
        plano = json.loads(texto)
    except:
        try:
            ultimo = max(texto.rfind("}"), texto.rfind("]"))
            texto_cortado = texto[:ultimo+1]
            abertas = texto_cortado.count("{") - texto_cortado.count("}")
            texto_cortado += "}" * abertas
            plano = json.loads(texto_cortado)
        except:
            plano = {
                "diagnostico": {"ja_sabe": "Perfil analisado.", "gaps_criticos": [], "gaps_complementares": []},
                "fases": [{"numero": 1, "titulo": "Plano", "objetivo": req.objetivo, "duracao": "A definir", "modulos": []}],
                "cronograma": None,
                "resultado_esperado": None
            }

    onboarding = {
        "ocupacao": req.ocupacao, "experiencia": req.experiencia,
        "nivel": req.nivel, "objetivo": req.objetivo,
        "tempo": req.tempo, "estilo": req.estilo
    }
    novo = salvar_plano_db(req.user_id, onboarding, plano)
    return {"ok": True, "plano_id": novo["id"], "plano": plano}

@app.post("/api/chat")
def chat(req: ChatRequest):
    contexto = buscar_conteudo(req.pergunta)

    system = f"""Você é um tutor de aprendizado personalizado da CEFIS.

PERFIL DO ALUNO:
- Nome: {req.user_name}
- Nível: {req.nivel}
- Estilo de aprendizagem: {req.estilo}
- Objetivo: {req.objetivo}

INSTRUÇÕES:
- Adapte a linguagem ao nível {req.nivel}
- Se o estilo for Visual: use tabelas e listas
- Se o estilo for Auditivo: use linguagem conversacional
- Se o estilo for Prático: foque em exemplos reais
- Fundamente com o conteúdo real da CEFIS abaixo

CONTEÚDO REAL DA PLATAFORMA CEFIS:
{contexto}"""

    mensagens = req.historico + [{"role": "user", "content": req.pergunta}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=system,
        messages=mensagens
    )
    return {"resposta": response.content[0].text}

@app.post("/api/conteudo")
def gerar_conteudo(req: ContentRequest):
    contexto = buscar_conteudo(req.tema, n=8)

    if req.tipo == "quiz":
        prompt = f"""Com base no conteúdo real da CEFIS abaixo, crie um questionário de 5 perguntas sobre "{req.tema}".
Nível: {req.nivel}
Conteúdo: {contexto}
Formato: 5 questões A/B/C/D, gabarito ao final, explicação de cada resposta."""
    else:
        prompt = f"""Com base no conteúdo real das aulas da CEFIS, crie uma apostila resumida sobre "{req.tema}".
Nível: {req.nivel}
Estilo: {req.estilo}
Conteúdo: {contexto}
Adapte ao estilo {req.estilo}. Inclua conceitos principais, exemplos e pontos de atenção."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"conteudo": response.content[0].text}

app.mount("/static", StaticFiles(directory="static"), name="static")


class AcompanhamentoRequest(BaseModel):
    user_id: int
    plano_id: str
    user_name: str
    objetivo: str
    nivel: str
    fases: list

@app.post("/api/acompanhamento")
def acompanhamento(req: AcompanhamentoRequest):
    dados = carregar_planos(req.user_id)
    plano = next((p for p in dados["planos"] if p["id"] == req.plano_id), None)
    if not plano:
        raise HTTPException(status_code=404, detail="Plano não encontrado")

    progresso_fases = plano.get("progresso_fases", {})
    fases_concluidas = list(progresso_fases.keys())
    
    # Acha a próxima fase não concluída
    todas_fases = req.fases
    proxima_fase = None
    for f in todas_fases:
        if str(f["numero"]) not in fases_concluidas:
            proxima_fase = f
            break

    if not proxima_fase:
        return {
            "mensagem": f"Parabéns, {req.user_name}! Você concluiu todas as fases do plano.",
            "proximo_passo": "Plano concluído",
            "fases_concluidas": len(fases_concluidas),
            "total_fases": len(todas_fases)
        }

    # Primeira visita
    if not plano.get("historico_acesso"):
        plano["historico_acesso"] = [datetime.now().strftime("%d/%m/%Y %H:%M")]
        with open(f"./dados_{req.user_id}.json", "w") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        return {
            "mensagem": f"Bem-vindo, {req.user_name}! Seu plano está pronto. Comece pela Fase 1.",
            "proximo_passo": proxima_fase["titulo"],
            "fases_concluidas": len(fases_concluidas),
            "total_fases": len(todas_fases)
        }

    # Visitas seguintes — mensagem baseada em progresso real
    pct = round((len(fases_concluidas) / len(todas_fases)) * 100) if todas_fases else 0
    
    if len(fases_concluidas) == 0:
        mensagem = f"Continue firme, {req.user_name}. Faça o quiz da Fase 1 para registrar seu progresso."
    else:
        mensagem = f"{len(fases_concluidas)} de {len(todas_fases)} fases concluídas ({pct}%). Próximo passo: {proxima_fase['titulo']}."

    return {
        "mensagem": mensagem,
        "proximo_passo": proxima_fase["titulo"],
        "fases_concluidas": len(fases_concluidas),
        "total_fases": len(todas_fases)
    }

class QuizFaseRequest(BaseModel):
    fase_titulo: str
    fase_objetivo: str
    modulos: list
    nivel: str

class SalvarProgressoRequest(BaseModel):
    user_id: int
    plano_id: str
    fase_numero: int
    fase_titulo: str
    acertos: int
    total: int

@app.post("/api/quiz-fase")
def quiz_fase(req: QuizFaseRequest):
    conteudo_fase = "\n".join([
        f"- {m.get('titulo', '')}: {m.get('conteudo', '')}"
        for m in req.modulos[:6]
    ])

    prompt = f"""Crie um quiz de 5 questões sobre a fase de estudos abaixo.

Fase: {req.fase_titulo}
Objetivo: {req.fase_objetivo}
Nível: {req.nivel}

Conteúdo estudado:
{conteudo_fase}

Retorne APENAS JSON válido neste formato:
{{
  "questoes": [
    {{
      "pergunta": "texto da pergunta",
      "opcoes": ["A) texto", "B) texto", "C) texto", "D) texto"],
      "correta": 0,
      "explicacao": "por que esta é a resposta correta"
    }}
  ]
}}

correta é o índice (0-3) da opção correta."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    texto = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(texto)
    except:
        raise HTTPException(status_code=500, detail="Erro ao gerar quiz")

@app.post("/api/salvar-progresso")
def salvar_progresso(req: SalvarProgressoRequest):
    dados = carregar_planos(req.user_id)
    plano = next((p for p in dados["planos"] if p["id"] == req.plano_id), None)
    if not plano:
        raise HTTPException(status_code=404, detail="Plano não encontrado")

    if "progresso_fases" not in plano:
        plano["progresso_fases"] = {}

    plano["progresso_fases"][str(req.fase_numero)] = {
        "fase_titulo": req.fase_titulo,
        "acertos": req.acertos,
        "total": req.total,
        "percentual": round((req.acertos / req.total) * 100),
        "concluido_em": datetime.now().strftime("%d/%m/%Y %H:%M")
    }

    # Calcula progresso geral
    total_fases = len(plano["plano"].get("fases", []))
    fases_concluidas = len(plano["progresso_fases"])
    plano["progresso_geral"] = round((fases_concluidas / total_fases) * 100) if total_fases > 0 else 0

    with open(f"./dados_{req.user_id}.json", "w") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    return {"ok": True, "progresso_geral": plano["progresso_geral"]}