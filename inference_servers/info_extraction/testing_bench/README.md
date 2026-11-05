# Testing Environment for Information Extraction from Emergency Call Transcriptions

## 1. Run the benchmarks

The following sequence of commands will install the required packages, load a specialized dataset from HF and make inferences with several models.

```sh
pip install -r requirements2.txt
python dataset_loading.py
python test_gliner.py results/gli_results.json
python test_azure.py results/azure_results.json
python test_gemini.py results/gemini_results.json
```

### 1.1. NuExtract

NuExtract needs a different environment for testing, which is a VLLM-based docker container

```sh
chmod +x nuextract/*.sh
cd nuextract && ./build_nuextract.sh
```

Then, run the container for NuExtract-2.0-2B:

```sh
./nuextract/up2b.sh
```
After the server is functioning, you can test this model.
```sh
python test_nuextract.py results/nuextract_results.json numind/NuExtract-2.0-2B
```

Now, do the same for 4B model.
```sh
./nuextract/up4b.sh
```

```sh
python test_nuextract.py results/nuextract_results.json numind/NuExtract-2.0-4B
```

### 1.2. Laya

O Laya responde só as seis classificações booleanas (não extrai entidades). O teste usa as mesmas perguntas do runner `laya` do backend e grava, além do Fmax, as métricas no limiar fixo usado em produção (`--threshold`, padrão 0.5).

```sh
pip install laya
python test_laya.py results/laya_results.json
# comparando o Fmax com resultados já gerados
python test_laya.py results/laya_results.json --compare results/serpro_open_results.json results/gemini_results.json
```

Opções úteis: `--limit N` para uma rodada rápida, `--device cpu|cuda`, `--batch-size`, `--max-len` e `--subfolder` (padrão `multilingual`). O arquivo gerado pode entrar no `make_df.py` junto com os demais; como não tem métrica de entidades, a coluna de qualidade geral fica vazia para ele.

## 2. Generate Results

```sh
python make_df.py results/nuextract_results.json results/gli_results.json results/azure_results.json results/gemini_results.json
python make_plots.py
```