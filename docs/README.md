# Construção da Página de Documentação (`docs/`)

Este diretório contém o gerador estático em Python que unifica todos os arquivos Markdown (`.md`) do repositório, os diagramas em `artefatos/` e as páginas interativas da API (`backend/docs/`) em uma única página estática (`docs/index.html`) pronta para publicação via **GitHub Pages**.

---

## 1. Configuração do Ambiente

Você pode preparar o ambiente usando **Conda / Mamba** (a partir do arquivo [`docs/env.yml`](env.yml)) ou instalar os pacotes diretamente via **`pip`**.

### Opção A: Usando Conda ou Mamba (`docs/env.yml`)

O arquivo [`docs/env.yml`](env.yml) já contém todas as dependências necessárias para criar o ambiente `hermes_webpage`:

**Com Mamba (recomendado):**
```bash
mamba env create -f docs/env.yml
mamba activate hermes_webpage
```

**Com Conda:**
```bash
conda env create -f docs/env.yml
conda activate hermes_webpage
```

### Opção B: Usando `pip`

Caso prefira usar `pip` diretamente (sem necessidade de criar um ambiente Conda/Mamba), certifique-se de ter uma versão compatível de **Python (>= 3.10)** e **`pip`** instalada e execute:

```bash
pip install markdown pymdown-extensions pygments pyyaml
```

## 2. Como Reconstruir a Documentação

A partir da raiz do repositório, execute o script [`docs/build_docs.py`](build_docs.py):

### Com o ambiente ativado

```bash
conda activate hermes_webpage
python docs/build_docs.py
```

### Ou chamando diretamente o executável do ambiente

```bash
python docs/build_docs.py
```

---

## 3. O que o Script Faz Automaticamente

Ao executar `docs/build_docs.py`, o script realiza as seguintes etapas:

1. **Leitura do Índice (`docs/markdowns_index.md`)**: Lê a lista oficial de arquivos `.md` e páginas da API definidos em [`docs/markdowns_index.md`](markdowns_index.md). Caso novos arquivos `.md` sejam adicionados ao repositório, o script também os detecta e inclui automaticamente na árvore.
2. **Sincronização da API Interativa (`docs/api/`)**: Copia automaticamente `swagger.html`, `openapi.html` e `openapi.json` de `backend/docs/` para `docs/api/`, garantindo que funcionem no GitHub Pages.
3. **Sincronização de Diagramas (`docs/assets/artefatos/`)**: Varre `artefatos/` em busca de diagramas (`.png`, `.svg`, `.jpg`), copia-os para `docs/assets/` e cria galerias visuais navegáveis.
4. **Resolução de Links Internos**: Converte links relativos entre os diferentes arquivos `.md` do repositório para navegação direta dentro da página única (`#page=...`).
5. **Montagem dos Templates (`docs/templates/`)**: Une os templates HTML, folhas de estilo CSS e scripts JavaScript para gerar o arquivo final `docs/index.html`.

---

## 4. Estrutura Modular e Personalização

Todo o código de apresentação (HTML, CSS e JS) fica separado da lógica Python dentro de `docs/templates/` para facilitar a personalização:

```text
docs/
├── build_docs.py            # Script Python que une templates, estilos e conteúdos
├── env.yml                  # Especificação do ambiente Conda/Mamba (hermes_webpage)
├── markdowns_index.md       # Índice de arquivos .md e páginas da API
├── README.md                # Este guia de construção da documentação
├── index.html               # Página estática gerada (output)
├── api/                     # Cópia sincronizada de backend/docs/ (output)
├── assets/                  # Cópia sincronizada dos diagramas de artefatos/ (output)
└── templates/               # Arquivos editáveis de layout, estilo e comportamento
    ├── base.html            # Estrutura principal da página (navbar, árvore lateral, palco, TOC)
    ├── styles.css           # Estilos principais e variáveis de cores (temas claro e escuro)
    ├── pygments.css         # Estilos de destaque de sintaxe para blocos de código
    ├── app.js               # Lógica de navegação, busca, árvore de arquivos e temas
    ├── tree_folder.html     # Template de pasta na árvore lateral
    ├── tree_file.html       # Template de arquivo na árvore lateral
    ├── openapi_spec.html    # Template da página de visualização do openapi.json
    ├── openapi_row.html     # Template de linha da tabela de endpoints
    ├── diagram_gallery.html # Template de galeria de diagramas
    └── diagram_card.html    # Template de card individual de diagrama
```

### Adicionando um novo documento Markdown

1. Crie o arquivo `.md` normalmente no diretório correspondente do projeto.
2. Adicione a entrada correspondente em [`docs/markdowns_index.md`](markdowns_index.md) para definir seu título amigável (opcional — se não adicionado, o script ainda o incluirá automaticamente usando o título do documento).
3. Execute `python docs/build_docs.py` para atualizar `docs/index.html`.

