import json
from json import JSONDecodeError
from random import random
import subprocess
from threading import Event, Thread
import time
from matplotlib import pyplot as plt


def runCommand(cmd, print_cmd=True):
    cmd = " ".join(cmd)
    # if print_cmd:
    #    print("\t> " + cmd)
    try:
        # Captura o output do comando
        result = subprocess.run(
            cmd, shell=True, text=True, capture_output=True, check=True
        )
        # print('Output:', result.stdout)  # Opcional: imprime o output capturado
        return result.stdout.strip()  # Retorna o output capturado
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o comando: {e}")
        print(f"Saída do erro: {e.stderr}")
        return None


def read_container_stats():
    cmd = ["docker", "compose", "stats", "--format", "json", "--no-stream"]
    raw_json = runCommand(cmd, print_cmd=True).split("\n")
    try:
        # print(raw_json)
        json_data = [json.loads(x) for x in raw_json]
        return json_data
    except JSONDecodeError as err:
        # print('Erro lendo docker stats')
        # print(err)
        return None


def watch_docker_stats(readings_vec: list, flag):
    print("Thread inside")
    waittime = 0.5

    while True:
        stats = read_container_stats()
        if stats != None:
            readings_vec.extend(stats)
        if flag.is_set():
            # print('Stoping thread')
            return None
        else:
            # print(readings)
            time.sleep(waittime)


def get_docker_measurer():
    flag = Event()
    flag.clear()
    readings = []
    t = Thread(target=watch_docker_stats, args=(readings, flag))

    return t, readings, flag


def read_audio_length(audio_info, audio_path):
    print("Lendo duracao de", audio_path)
    try:
        import soundfile as sf

        y, sr = sf.read(audio_path)
        duration_seconds = len(y) / sr
        print(
            f"Duração real do áudio: {duration_seconds:.2f} segundos (sample rate: {sr}Hz)"
        )
    except ImportError:
        # Fallback para metadata se nenhuma biblioteca de áudio estiver disponível
        print("Aviso: Nenhuma biblioteca de áudio disponível, usando metadata")
        duration_minutes = audio_info["Emergencia"]["Duração da Ligacao (Minutos)"]
        duration_seconds = duration_minutes * 60
        print(f"Duração do metadata: {duration_seconds:.2f} segundos")

    audio_info["Duracao Real"] = duration_seconds


def calcular_delays(audio_info, carga_de_ligacoes):
    """Calcula os delays de início de cada áudio, baseado na carga de ligações"""
    total_audio_time = sum(
        a["Duracao Real"] for a in audio_info
    )  # soma total das durações
    tempo_simulacao = (
        total_audio_time / carga_de_ligacoes
    )  # quanto tempo 'real' vamos levar
    tempo_minimo_simulacao = max(
        a["Duracao Real"] for a in audio_info
    )  # no mínimo, o maior áudio
    menor_audio = min(a["Duracao Real"] for a in audio_info)
    print("Maior audio:", tempo_minimo_simulacao)
    if tempo_simulacao < tempo_minimo_simulacao:
        print(
            f"Atenção: carga_de_ligacoes={carga_de_ligacoes} é muito alta para {len(audio_info)} áudios. Ajustando para {total_audio_time/tempo_minimo_simulacao:.1f}"
        )
        carga_de_ligacoes = total_audio_time / tempo_minimo_simulacao
        tempo_simulacao = tempo_minimo_simulacao
        print(
            f"Novo carga_de_ligacoes={carga_de_ligacoes:.1f}, tempo_simulacao={tempo_simulacao:.1f}s"
        )

    free_seconds = [tempo_simulacao - a["Duracao Real"] for a in audio_info]
    delay_factor = [random() for _ in audio_info]
    delays_vec = [fs * df for fs, df in zip(free_seconds, delay_factor)]

    delays2 = []
    for a, delay in zip(audio_info, delays_vec):
        if a["Duracao Real"] == menor_audio:
            a["Delay"] = 0.0
        elif a["Duracao Real"] == tempo_minimo_simulacao:
            a["Delay"] = tempo_simulacao - a["Duracao Real"]
        else:
            a["Delay"] = delay
        delays2.append(a["Delay"])
    return delays2


def calcular_delay_hf(audio_dataset, carga_de_ligacoes: float, test_path):
    """Calcula os delays de início de cada áudio, baseado na carga de ligações"""
    duracoes = []
    for audio in audio_dataset:
        duracoes.append(
            audio["audio"]["array"].shape[0] / audio["audio"]["sampling_rate"]
        )
    total_audio_time = sum(duracoes)  # soma total das durações
    tempo_simulacao = (
        total_audio_time / carga_de_ligacoes
    )  # quanto tempo 'real' vamos levar
    tempo_minimo_simulacao = max(duracoes)  # no mínimo, o maior áudio
    menor_audio = min(duracoes)
    print("Maior audio:", tempo_minimo_simulacao)
    if tempo_simulacao < tempo_minimo_simulacao:
        print(
            f"Atenção: carga_de_ligacoes={carga_de_ligacoes} é muito alta para {len(audio_dataset)} áudios. Ajustando para {total_audio_time/tempo_minimo_simulacao:.1f}"
        )
        carga_de_ligacoes = total_audio_time / tempo_minimo_simulacao
        tempo_simulacao = tempo_minimo_simulacao
        print(
            f"Novo carga_de_ligacoes={carga_de_ligacoes:.1f}, tempo_simulacao={tempo_simulacao:.1f}s"
        )

    print(f"Duração total do teste: {tempo_simulacao:.1f}s")
    print(f"Carga de ligações: {carga_de_ligacoes:.1f}")
    print(f"Número de áudios: {len(audio_dataset)}")
    print(f"Menor áudio: {menor_audio:.1f}s")
    print(f"Maior áudio: {tempo_minimo_simulacao:.1f}s")
    print(f"Total de áudio: {total_audio_time:.1f}s")

    import numpy as np

    free_seconds = [tempo_simulacao - d for d in duracoes]
    n_audios = len(audio_dataset)
    n_linear = int(n_audios * 0.9)
    n_random = n_audios - n_linear

    # 75% recebem uma distribuição linear (monótona) de 0 a 1, com um leve ruído (desvio padrão de 0.05)
    linear_factors = np.linspace(0, 1, n_linear)
    random_factors = np.random.rand(n_random)
    all_indices = np.random.permutation(n_audios)
    # Ordenamos os índices lineares para manter a rampa monótona crescente ao longo do dataset
    linear_indices = sorted(all_indices[:n_linear])
    random_indices = all_indices[n_linear:]

    delay_factors = np.zeros(n_audios)
    delay_factors[linear_indices] = linear_factors
    delay_factors[random_indices] = random_factors

    delays_vec = [fs * df for fs, df in zip(free_seconds, delay_factors)]
    """delay_factor = [random() for _ in audio_dataset]
    delays_vec = [fs * df for fs, df in zip(free_seconds, delay_factor)]"""

    delays2 = []
    for duracao, delay in zip(duracoes, delays_vec):
        if duracao == menor_audio:
            delay = 0.0
        elif duracao == tempo_minimo_simulacao:
            delay = tempo_simulacao - duracao
        delays2.append(delay)

    # Adicionar duracoes e delays2 como novas colunas
    audio_dataset = audio_dataset.add_column("duration", duracoes)
    audio_dataset = audio_dataset.add_column("delay", delays2)

    for row in audio_dataset:
        print(
            f"Audio começa em {row['delay']:.1f}s e termina em {row['delay'] + row['duration']:.1f}s"
        )

    starts = []
    ends = []
    for row in audio_dataset:
        print(
            f"Áudio {row['ID']} -> duração {row['duration']:.1f}s, delay {row['delay']:.1f}s"
        )
        starts.append(row["delay"])
        ends.append(row["delay"] + row["duration"])

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    start_ends = list(zip(starts, ends))
    bin_width = 5
    bin_edges = np.arange(0, max(ends), bin_width)
    counts = []
    for i in range(len(bin_edges) - 1):
        bin_s = bin_edges[i]
        bin_e = bin_edges[i + 1]

        count = 0
        for s, e in start_ends:
            if (s >= bin_s and s <= bin_e) or (e >= bin_s and e <= bin_e):
                count += 1
            elif bin_s >= s and bin_s <= e:
                count += 1
        counts.append(count)
    bin_middles = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]
    ax.plot(bin_middles, counts, c="#373d41ff", linewidth=7)
    # ax.set_xlabel('Segundos Após Início do Teste')
    ax.set_ylabel("Chamadas Concorrentes")
    ax.set_title("Agenda de Ligações")
    fig.tight_layout()
    fig.savefig(test_path + "/original_call_delays.png", dpi=120)

    return audio_dataset
