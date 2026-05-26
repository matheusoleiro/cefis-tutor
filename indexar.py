import os
import re
import chromadb
from sentence_transformers import SentenceTransformer

# Configuração
PATH = "./output"
CHROMA_PATH = "./chroma_db"

# Inicializa o modelo de embeddings e o banco vetorial
print("Carregando modelo de embeddings...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("cefis")

def parse_vtt(filepath):
    """Extrai só o texto do arquivo .vtt, remove timestamps"""
    with open(filepath, "r") as f:
        content = f.read()
    linhas = content.split("\n")
    textos = []
    for linha in linhas:
        linha = linha.strip()
        if not linha or linha == "WEBVTT":
            continue
        if re.match(r"^\d+$", linha):
            continue
        if re.match(r"[\d:.]+ --> [\d:.]+", linha):
            continue
        textos.append(linha)
    return " ".join(textos)

def chunk_texto(texto, tamanho=500, overlap=50):
    """Divide o texto em chunks com overlap"""
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho])
        chunks.append(chunk)
        i += tamanho - overlap
    return chunks

# Indexa tudo
subpastas = [f for f in os.listdir(PATH) if os.path.isdir(f"{PATH}/{f}")]
print(f"Total de cursos: {len(subpastas)}")

total_chunks = 0
for idx, curso_id in enumerate(subpastas):
    curso_path = f"{PATH}/{curso_id}"
    
    # Pega o título do curso
    import json
    details_path = f"{curso_path}/details.json"
    curso_titulo = curso_id
    if os.path.exists(details_path):
        with open(details_path, "r") as f:
            details = json.load(f)
            curso_titulo = details.get("data", {}).get("title", curso_id)

    lessons_path = f"{curso_path}/lessons"
    if not os.path.exists(lessons_path):
        continue

    for aula_id in os.listdir(lessons_path):
        aula_path = f"{lessons_path}/{aula_id}"
        
        # Pega título da aula
        aula_titulo = aula_id
        aula_details = f"{aula_path}/details.json"
        if os.path.exists(aula_details):
            with open(aula_details, "r") as f:
                ad = json.load(f)
                aula_titulo = ad.get("title", aula_id)

        # Procura o .vtt
        vtt_file = None
        for arquivo in os.listdir(aula_path):
            if arquivo.endswith(".vtt"):
                vtt_file = f"{aula_path}/{arquivo}"
                break
        
        if not vtt_file:
            continue

        texto = parse_vtt(vtt_file)
        if not texto.strip():
            continue

        chunks = chunk_texto(texto)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{curso_id}_{aula_id}_{i}"
            embedding = model.encode(chunk).tolist()
            
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "curso_id": curso_id,
                    "curso_titulo": curso_titulo,
                    "aula_id": aula_id,
                    "aula_titulo": aula_titulo,
                    "chunk_index": i
                }]
            )
            total_chunks += 1

    if idx % 50 == 0:
        print(f"Progresso: {idx}/{len(subpastas)} cursos | {total_chunks} chunks indexados")

print(f"\nConcluído. Total de chunks: {total_chunks}")
