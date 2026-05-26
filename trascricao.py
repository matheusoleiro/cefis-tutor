import os

path = "./output"
subpastas = [f for f in os.listdir(path) if os.path.isdir(f"{path}/{f}")]

# Acha o primeiro .vtt que existir
for curso in subpastas:
    lessons_path = f"{path}/{curso}/lessons"
    if not os.path.exists(lessons_path):
        continue
    for aula in os.listdir(lessons_path):
        aula_path = f"{lessons_path}/{aula}"
        for arquivo in os.listdir(aula_path):
            if arquivo.endswith(".vtt"):
                with open(f"{aula_path}/{arquivo}", "r") as f:
                    print(f.read()[:2000])
                exit()