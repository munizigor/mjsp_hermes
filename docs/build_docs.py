#!/home/pita/fs/miniforge3/envs/hermes_webpage/bin/python
"""
Script de construção da documentação estática unificada do projeto Hermes.
Une os templates HTML, folhas de estilo CSS e scripts JS em `docs/templates/`
com os arquivos Markdown listados em `docs/markdowns_index.md` e a documentação
interativa da API (`backend/docs/`), gerando `docs/index.html`.

Uso:
    /home/pita/fs/miniforge3/envs/hermes_webpage/bin/python docs/build_docs.py
"""

from __future__ import annotations

import html
import json
import posixpath
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
from pygments.formatters import HtmlFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
TEMPLATES_DIR = DOCS_DIR / "templates"
INDEX_MD_PATH = DOCS_DIR / "markdowns_index.md"
OUTPUT_HTML_PATH = DOCS_DIR / "index.html"
API_OUT_DIR = DOCS_DIR / "api"
ASSETS_OUT_DIR = DOCS_DIR / "assets"

EXCLUDED_DIRS = {
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "staticfiles",
    "__pycache__",
}

EXCLUDED_FILES = {
    "docs/markdowns_index.md",
}


def load_template(name: str) -> str:
    """Lê um arquivo de template/recurso dentro de `docs/templates/`."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def render_template(template_str: str, **context: str) -> str:
    """Substitui placeholders `{{CHAVE}}` no template pelos valores fornecidos."""
    result = template_str
    for key, value in context.items():
        result = result.replace(f"{{{{{key}}}}}", str(value))
    return result


def ensure_pygments_css() -> str:
    """
    Lê `docs/templates/pygments.css` (ou gera o arquivo padrão na primeira execução
    para permitir fácil personalização posterior).
    """
    pygments_css_path = TEMPLATES_DIR / "pygments.css"
    if not pygments_css_path.exists():
        light_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
        dark_css = HtmlFormatter(style="monokai").get_style_defs(
            '[data-theme="dark"] .codehilite'
        )
        pygments_css_path.write_text(
            f"/* Pygments Light Theme */\n{light_css}\n\n/* Pygments Dark Theme */\n{dark_css}\n",
            encoding="utf-8",
        )
    return pygments_css_path.read_text(encoding="utf-8")


def parse_markdowns_index(index_path: Path) -> list[dict[str, str]]:
    """Lê `docs/markdowns_index.md` e extrai a lista de arquivos e seus títulos."""
    entries: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        link_pattern = re.compile(r"-\s+\[([^\]]+)\]\(([^)]+)\)")
        for match in link_pattern.finditer(content):
            raw_label = match.group(1).strip()
            raw_target = match.group(2).strip()
            resolved = (index_path.parent / raw_target).resolve()
            try:
                rel_path = resolved.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue

            if not resolved.exists() or rel_path in seen_paths:
                continue

            clean_label = re.sub(r"\s*\(`[^`]+`\)\s*$", "", raw_label).strip()
            clean_label = re.sub(r"\s*\([^)]+\.md\)\s*$", "", clean_label).strip()

            entries.append(
                {
                    "path": rel_path,
                    "title": clean_label or Path(rel_path).name,
                }
            )
            seen_paths.add(rel_path)

    for md_file in sorted(REPO_ROOT.rglob("*.md")):
        rel_parts = md_file.relative_to(REPO_ROOT).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        rel_posix = md_file.relative_to(REPO_ROOT).as_posix()
        if rel_posix in EXCLUDED_FILES or rel_posix in seen_paths:
            continue
        print(f"[INFO] Arquivo .md adicional detectado e incluído: {rel_posix}")
        entries.append(
            {
                "path": rel_posix,
                "title": (
                    md_file.stem
                    if md_file.name.lower() != "readme.md"
                    else f"README ({md_file.parent.name or 'root'})"
                ),
            }
        )
        seen_paths.add(rel_posix)

    return entries


def sync_api_docs() -> None:
    """Copia os arquivos interativos de `backend/docs/` (exceto openapi.html) para `docs/api/`."""
    backend_docs = REPO_ROOT / "backend" / "docs"
    if not backend_docs.exists():
        return
    if API_OUT_DIR.exists():
        shutil.rmtree(API_OUT_DIR)
    API_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in sorted(backend_docs.iterdir()):
        if item.name.lower() == "openapi.html":
            continue
        if item.is_file() and item.suffix.lower() in {".html", ".json", ".yaml", ".yml"}:
            shutil.copy2(item, API_OUT_DIR / item.name)


def sync_artefatos_diagrams() -> list[dict[str, Any]]:
    """
    Descobre e copia automaticamente diagramas (.png, .svg, .jpg, .jpeg) dentro de
    `artefatos/` para `docs/assets/artefatos/` e gera páginas de galeria usando os templates HTML.
    """
    artefatos_root = REPO_ROOT / "artefatos"
    if not artefatos_root.exists():
        return []

    if ASSETS_OUT_DIR.exists():
        shutil.rmtree(ASSETS_OUT_DIR)

    known_titles = {
        "artefatos/diagramas/diagrama_arquitetural": "Diagramas Arquiteturais",
        "artefatos/diagramas/diagrama_sequencia": "Diagramas de Sequência",
        "artefatos/fluxogramas/fluxograma_arquitetural": "Fluxograma Arquitetural",
        "artefatos/fluxogramas/hermes_atendimento": "Fluxogramas — Atendimento",
        "artefatos/fluxogramas/hermes_funcionalidades": "Fluxogramas — Funcionalidades",
    }

    card_tpl = load_template("diagram_card.html")
    gallery_tpl = load_template("diagram_gallery.html")
    virtual_pages: list[dict[str, Any]] = []

    # Descobre qualquer subdiretório em `artefatos/` que contenha imagens
    candidate_dirs = sorted(
        {f.parent for f in artefatos_root.rglob("*") if f.is_file() and f.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}}
    )

    for src_dir in candidate_dirs:
        rel_dir = src_dir.relative_to(REPO_ROOT).as_posix()
        title = known_titles.get(
            rel_dir,
            src_dir.name.replace("_", " ").replace("-", " ").title(),
        )

        images = sorted(
            f
            for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}
        )
        if not images:
            continue

        dest_dir = ASSETS_OUT_DIR / rel_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        cards_html: list[str] = []
        for img in images:
            shutil.copy2(img, dest_dir / img.name)
            asset_rel = f"assets/{rel_dir}/{img.name}"
            cards_html.append(
                render_template(
                    card_tpl,
                    IMAGE_NAME=html.escape(img.name),
                    ASSET_REL=html.escape(asset_rel),
                )
            )

        body_html = render_template(
            gallery_tpl,
            TITLE=html.escape(title),
            REL_DIR=html.escape(rel_dir),
            CARDS_HTML="".join(cards_html),
        )

        virtual_pages.append(
            {
                "id": rel_dir,
                "path": rel_dir,
                "name": Path(rel_dir).name,
                "title": title,
                "kind": "gallery",
                "html": body_html,
                "toc": [],
                "search_text": f"{title} {rel_dir} " + " ".join(img.name for img in images),
            }
        )

    return virtual_pages


def extract_first_heading(md_text: str, fallback: str) -> str:
    """Extrai o primeiro H1 do Markdown caso exista."""
    for line in md_text.splitlines():
        m = re.match(r"^#\s+(.+)$", line.strip())
        if m:
            return re.sub(r"[`*_]", "", m.group(1)).strip()
    return fallback


def convert_markdown_to_html(
    md_text: str, current_rel_path: str, known_page_ids: set[str]
) -> tuple[str, list[dict[str, Any]], str]:
    """
    Converte conteúdo Markdown em HTML e reescreve links internos entre arquivos.
    """
    md = markdown.Markdown(
        extensions=[
            "extra",
            "tables",
            "sane_lists",
            "toc",
            "pymdownx.superfences",
            "pymdownx.highlight",
            "pymdownx.tasklist",
        ],
        extension_configs={
            "toc": {
                "permalink": False,
                "toc_depth": "2-3",
            },
            "pymdownx.highlight": {
                "use_pygments": True,
                "guess_lang": False,
                "css_class": "codehilite",
            },
        },
    )

    raw_html = md.convert(md_text)
    toc_tokens = getattr(md, "toc_tokens", [])
    current_dir = posixpath.dirname(current_rel_path)

    def rewrite_href(match: re.Match[str]) -> str:
        prefix = match.group(1)
        href = match.group(2).strip()
        suffix = match.group(3)

        if (
            href.startswith(("http://", "https://", "mailto:", "#", "javascript:"))
            or not href
        ):
            if href.startswith(("http://", "https://")):
                return f'{prefix}{href}{suffix} target="_blank" rel="noopener"'
            return match.group(0)

        if "#" in href:
            path_part, anchor_part = href.split("#", 1)
        else:
            path_part, anchor_part = href, ""

        norm_target = posixpath.normpath(posixpath.join(current_dir, path_part)).lstrip("./")

        for cand in (norm_target, posixpath.join(norm_target, "README.md")):
            if cand in known_page_ids:
                new_href = f"#page={cand}"
                if anchor_part:
                    new_href += f"&section={anchor_part}"
                return f"{prefix}{new_href}{suffix}"

        return match.group(0)

    rewritten_html = re.sub(r'(href=["\'])([^"\']+)(["\'])', rewrite_href, raw_html)
    plain_text = re.sub(r"<[^>]+>", " ", rewritten_html)
    plain_text = re.sub(r"\s+", " ", html.unescape(plain_text)).strip()

    return rewritten_html, toc_tokens, plain_text


def build_openapi_json_page(rel_path: str, title: str) -> dict[str, Any]:
    """Cria a página de visualização para `backend/docs/openapi.json` usando templates."""
    json_path = REPO_ROOT / rel_path
    spec = json.loads(json_path.read_text(encoding="utf-8"))
    info = spec.get("info", {})
    paths = spec.get("paths", {})

    row_tpl = load_template("openapi_row.html")
    spec_tpl = load_template("openapi_spec.html")

    rows_html: list[str] = []
    for endpoint, methods in paths.items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            rows_html.append(
                render_template(
                    row_tpl,
                    BADGE_CLASS=f"method-badge method-{html.escape(method.lower())}",
                    METHOD_UPPER=html.escape(method.upper()),
                    ENDPOINT=html.escape(endpoint),
                    SUMMARY=html.escape(details.get("summary", "")),
                    OP_ID=html.escape(details.get("operationId", "")),
                )
            )

    pretty_json = html.escape(json.dumps(spec, indent=2, ensure_ascii=False))
    body_html = render_template(
        spec_tpl,
        TITLE=html.escape(title),
        OPENAPI_VERSION=html.escape(str(spec.get("openapi", "3.1.0"))),
        API_TITLE=html.escape(str(info.get("title", "Hermes API"))),
        API_VERSION=html.escape(str(info.get("version", "0.1.0"))),
        ENDPOINT_COUNT=str(len(rows_html)),
        ROWS_HTML="".join(rows_html),
        PRETTY_JSON=pretty_json,
    )

    return {
        "id": rel_path,
        "path": rel_path,
        "name": Path(rel_path).name,
        "title": title,
        "kind": "openapi_json",
        "html": body_html,
        "toc": [
            {
                "id": "endpoints-resumo",
                "name": f"Resumo de Endpoints ({len(rows_html)})",
                "children": [],
            },
            {"id": "json-completo", "name": "Especificação JSON Completa", "children": []},
        ],
        "search_text": f"{title} {rel_path} openapi json " + " ".join(paths.keys()),
    }


def build_file_tree(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Organiza a lista de páginas em uma estrutura de árvore de diretórios."""
    root: dict[str, Any] = {"name": "mjsp_hermes", "path": "", "dirs": {}, "files": []}

    for page in pages:
        parts = page["path"].split("/")
        node = root
        cur_path_parts: list[str] = []
        for folder in parts[:-1]:
            cur_path_parts.append(folder)
            cur_path = "/".join(cur_path_parts)
            if folder not in node["dirs"]:
                node["dirs"][folder] = {
                    "name": folder,
                    "path": cur_path,
                    "dirs": {},
                    "files": [],
                }
            node = node["dirs"][folder]
        node["files"].append(page)

    return root


def render_tree_html(
    node: dict[str, Any], folder_tpl: str, file_tpl: str, depth: int = 0
) -> str:
    """Renderiza a árvore de diretórios e arquivos usando `tree_folder.html` e `tree_file.html`."""
    items_html: list[str] = []

    if depth == 0:
        for file_page in node["files"]:
            items_html.append(render_file_item_html(file_page, file_tpl))

    for folder_name in sorted(node["dirs"].keys()):
        child_node = node["dirs"][folder_name]
        children_html = render_tree_html(child_node, folder_tpl, file_tpl, depth + 1)
        is_api_dir = child_node["path"] == "backend/docs"
        items_html.append(
            render_template(
                folder_tpl,
                FOLDER_PATH=html.escape(child_node["path"]),
                FOLDER_NAME=html.escape(folder_name),
                BADGE_CLASS="api-badge" if is_api_dir else "",
                BADGE_TEXT="API" if is_api_dir else "",
                CHILDREN_HTML=children_html,
            )
        )

    if depth > 0:
        sorted_files = sorted(
            node["files"],
            key=lambda p: (0 if p["name"].lower() == "readme.md" else 1, p["name"].lower()),
        )
        for file_page in sorted_files:
            items_html.append(render_file_item_html(file_page, file_tpl))

    return "\n".join(items_html)


def render_file_item_html(page: dict[str, Any], file_tpl: str) -> str:
    """Renderiza um item de arquivo na árvore lateral usando `tree_file.html`."""
    kind = page["kind"]
    if kind == "api_iframe":
        icon = "⚡"
        badge_class = "interactive-badge"
        badge_text = "Interativo"
    elif kind == "openapi_json":
        icon = "📋"
        badge_class = "json-badge"
        badge_text = "JSON"
    elif kind == "gallery":
        icon = "🖼️"
        badge_class = ""
        badge_text = ""
    else:
        icon = "📄"
        badge_class = ""
        badge_text = ""

    return render_template(
        file_tpl,
        PAGE_ID=html.escape(page["id"]),
        PAGE_PATH=html.escape(page["path"]),
        PAGE_TITLE=html.escape(page["title"]),
        ICON=icon,
        PAGE_NAME=html.escape(page["name"]),
        BADGE_CLASS=badge_class,
        BADGE_TEXT=badge_text,
    )


def generate_html() -> None:
    """Executa o pipeline de compilação unindo templates, CSS, JS e conteúdos."""
    sync_api_docs()
    virtual_diagram_pages = sync_artefatos_diagrams()
    entries = parse_markdowns_index(INDEX_MD_PATH)

    known_page_ids: set[str] = {e["path"] for e in entries} | {
        vp["id"] for vp in virtual_diagram_pages
    }

    pages: list[dict[str, Any]] = []

    for entry in entries:
        rel_path = entry["path"]
        title = entry["title"]
        full_path = REPO_ROOT / rel_path
        suffix = full_path.suffix.lower()

        if suffix == ".md":
            md_text = full_path.read_text(encoding="utf-8")
            doc_title = title or extract_first_heading(md_text, full_path.name)
            rendered_html, toc_tokens, plain_text = convert_markdown_to_html(
                md_text, rel_path, known_page_ids
            )
            pages.append(
                {
                    "id": rel_path,
                    "path": rel_path,
                    "name": full_path.name,
                    "title": doc_title,
                    "kind": "markdown",
                    "html": rendered_html,
                    "toc": toc_tokens,
                    "search_text": f"{doc_title} {rel_path} {plain_text}",
                }
            )
        elif suffix == ".html" and rel_path.startswith("backend/docs/"):
            api_file_name = full_path.name
            if api_file_name.lower() == "openapi.html":
                continue
            pages.append(
                {
                    "id": rel_path,
                    "path": rel_path,
                    "name": api_file_name,
                    "title": title,
                    "kind": "api_iframe",
                    "iframe_src": f"api/{api_file_name}",
                    "html": "",
                    "toc": [],
                    "search_text": f"{title} {rel_path} API REST FastAPI Swagger OpenAPI endpoints",
                }
            )
        elif suffix == ".json" and rel_path.startswith("backend/docs/"):
            pages.append(build_openapi_json_page(rel_path, title))

    pages.extend(virtual_diagram_pages)

    folder_tpl = load_template("tree_folder.html")
    file_tpl = load_template("tree_file.html")
    base_tpl = load_template("base.html")
    styles_css = load_template("styles.css")
    pygments_css = ensure_pygments_css()
    app_js = load_template("app.js")

    tree = build_file_tree(pages)
    tree_html = render_tree_html(tree, folder_tpl, file_tpl)

    html_doc = render_template(
        base_tpl,
        STYLES_CSS=styles_css,
        PYGMENTS_CSS=pygments_css,
        TREE_HTML=tree_html,
        PAGES_JSON=json.dumps(pages, ensure_ascii=False),
        BUILD_TIMESTAMP=datetime.now().strftime("%d/%m/%Y %H:%M"),
        APP_JS=app_js,
    )

    OUTPUT_HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"[OK] Documentação gerada com sucesso em: {OUTPUT_HTML_PATH}")
    print(f"[OK] Total de páginas indexadas na árvore: {len(pages)}")


if __name__ == "__main__":
    generate_html()
