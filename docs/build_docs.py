#!/home/pita/fs/miniforge3/envs/hermes_webpage/bin/python
"""
Script de construção da documentação estática unificada do projeto Hermes.
Gera `docs/index.html` a partir dos arquivos listados em `docs/markdowns_index.md`
(e descobre novos arquivos `.md` no repositório), além de sincronizar a documentação
interativa da API (`backend/docs/`) para `docs/api/`.

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


def parse_markdowns_index(index_path: Path) -> list[dict[str, str]]:
    """Lê `docs/markdowns_index.md` e extrai a lista de arquivos e seus títulos."""
    entries: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        # Procura links markdown no formato: - [Título](../caminho/arquivo.md)
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

            # Remove `(caminho/arquivo)` do final do label se existir para ficar limpo
            clean_label = re.sub(r"\s*\(`[^`]+`\)\s*$", "", raw_label).strip()
            clean_label = re.sub(r"\s*\([^)]+\.md\)\s*$", "", clean_label).strip()

            entries.append(
                {
                    "path": rel_path,
                    "title": clean_label or Path(rel_path).name,
                }
            )
            seen_paths.add(rel_path)

    # Descobre automaticamente outros arquivos .md no repositório que não estejam excluídos
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
                "title": md_file.stem if md_file.name.lower() != "readme.md" else f"README ({md_file.parent.name or 'root'})",
            }
        )
        seen_paths.add(rel_posix)

    return entries


def sync_api_docs() -> None:
    """Copia os arquivos interativos de `backend/docs/` para `docs/api/`."""
    backend_docs = REPO_ROOT / "backend" / "docs"
    if not backend_docs.exists():
        return
    API_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in backend_docs.iterdir():
        if item.is_file() and item.suffix.lower() in {".html", ".json", ".yaml", ".yml"}:
            shutil.copy2(item, API_OUT_DIR / item.name)


def sync_artefatos_diagrams() -> list[dict[str, Any]]:
    """
    Copia diagramas (.png, .svg) de `artefatos/` para `docs/assets/artefatos/`
    e gera páginas de galeria virtuais para que os links em `artefatos/README.md`
    funcionem diretamente dentro da página estática.
    """
    diagram_dirs = [
        ("artefatos/diagramas/diagrama_arquitetural", "Diagramas Arquiteturais"),
        ("artefatos/diagramas/diagrama_sequencia", "Diagramas de Sequência"),
        ("artefatos/fluxogramas/fluxograma_arquitetural", "Fluxograma Arquitetural"),
        ("artefatos/fluxogramas/hermes_atendimento", "Fluxogramas — Atendimento"),
        ("artefatos/fluxogramas/hermes_funcionalidades", "Fluxogramas — Funcionalidades"),
    ]
    virtual_pages: list[dict[str, Any]] = []

    for rel_dir, title in diagram_dirs:
        src_dir = REPO_ROOT / rel_dir
        if not src_dir.exists() or not src_dir.is_dir():
            continue

        images = sorted(
            [
                f
                for f in src_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}
            ]
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
                f"""
                <div class="diagram-card">
                    <div class="diagram-card-header">
                        <span class="diagram-filename">{html.escape(img.name)}</span>
                        <a href="{html.escape(asset_rel)}" target="_blank" rel="noopener" class="btn-sm">Abrir original ↗</a>
                    </div>
                    <div class="diagram-img-wrap">
                        <a href="{html.escape(asset_rel)}" target="_blank" rel="noopener">
                            <img src="{html.escape(asset_rel)}" alt="{html.escape(img.name)}" loading="lazy" />
                        </a>
                    </div>
                </div>
                """
            )

        body_html = f"""
        <h1>{html.escape(title)}</h1>
        <p>Artefatos visuais localizados em <code>{html.escape(rel_dir)}/</code>. Clique em qualquer diagrama para ampliá-lo em uma nova aba.</p>
        <p><a href="#page=artefatos/README.md">← Voltar para Artefatos do Projeto (<code>artefatos/README.md</code>)</a></p>
        <div class="diagram-gallery">
            {''.join(cards_html)}
        </div>
        """

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
    Converte conteúdo Markdown em HTML com suporte a GFM, tabelas, código colorido
    e reescrita de links internos entre os documentos do repositório.
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

        # Mantém links externos, âncoras puras ou mailto
        if (
            href.startswith(("http://", "https://", "mailto:", "#", "javascript:"))
            or not href
        ):
            if href.startswith(("http://", "https://")):
                return f'{prefix}{href}{suffix} target="_blank" rel="noopener"'
            return match.group(0)

        # Separa caminho e eventual âncora (#secao)
        if "#" in href:
            path_part, anchor_part = href.split("#", 1)
        else:
            path_part, anchor_part = href, ""

        norm_target = posixpath.normpath(posixpath.join(current_dir, path_part)).lstrip("./")

        # Verifica se aponta para uma página conhecida (arquivo .md, api html/json ou galeria de diagramas)
        candidates = [
            norm_target,
            posixpath.join(norm_target, "README.md"),
        ]
        for cand in candidates:
            if cand in known_page_ids:
                new_href = f"#page={cand}"
                if anchor_part:
                    new_href += f"&section={anchor_part}"
                return f"{prefix}{new_href}{suffix}"

        return match.group(0)

    rewritten_html = re.sub(r'(href=["\'])([^"\']+)(["\'])', rewrite_href, raw_html)

    # Extrai texto plano para busca rápida no cliente
    plain_text = re.sub(r"<[^>]+>", " ", rewritten_html)
    plain_text = re.sub(r"\s+", " ", html.unescape(plain_text)).strip()

    return rewritten_html, toc_tokens, plain_text


def build_openapi_json_page(rel_path: str, title: str) -> dict[str, Any]:
    """Cria uma página de visualização para o `backend/docs/openapi.json`."""
    json_path = REPO_ROOT / rel_path
    spec = json.loads(json_path.read_text(encoding="utf-8"))
    info = spec.get("info", {})
    paths = spec.get("paths", {})

    rows_html: list[str] = []
    for endpoint, methods in paths.items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            summary = details.get("summary", "")
            op_id = details.get("operationId", "")
            badge_class = f"method-badge method-{method.lower()}"
            rows_html.append(
                f"""
                <tr>
                    <td><span class="{badge_class}">{html.escape(method.upper())}</span></td>
                    <td><code>{html.escape(endpoint)}</code></td>
                    <td>{html.escape(summary)}</td>
                    <td><small class="text-muted"><code>{html.escape(op_id)}</code></small></td>
                </tr>
                """
            )

    pretty_json = html.escape(json.dumps(spec, indent=2, ensure_ascii=False))
    body_html = f"""
    <div class="api-spec-header">
        <h1>{html.escape(title)}</h1>
        <p>Especificação OpenAPI <code>{html.escape(str(spec.get('openapi', '3.1.0')))}</code> — <strong>{html.escape(str(info.get('title', 'Hermes API')))}</strong> (v{html.escape(str(info.get('version', '0.1.0')))}).</p>
        <div class="api-action-bar">
            <a href="#page=backend/docs/swagger.html" class="btn-primary">⚡ Abrir no Swagger / Scalar</a>
            <a href="#page=backend/docs/openapi.html" class="btn-secondary">📘 Abrir no ReDoc</a>
            <a href="api/openapi.json" target="_blank" rel="noopener" class="btn-secondary" download="openapi.json">⬇ Baixar openapi.json</a>
        </div>
    </div>
    <h2 id="endpoints-resumo">Resumo de Endpoints ({len(rows_html)})</h2>
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Método</th>
                    <th>Endpoint</th>
                    <th>Resumo</th>
                    <th>Operation ID</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    <h2 id="json-completo">Especificação JSON Completa</h2>
    <pre><code class="language-json">{pretty_json}</code></pre>
    """

    return {
        "id": rel_path,
        "path": rel_path,
        "name": Path(rel_path).name,
        "title": title,
        "kind": "openapi_json",
        "html": body_html,
        "toc": [
            {"id": "endpoints-resumo", "name": f"Resumo de Endpoints ({len(rows_html)})", "children": []},
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


def render_tree_html(node: dict[str, Any], depth: int = 0) -> str:
    """Renderiza a árvore de diretórios e arquivos como HTML para a barra lateral."""
    items_html: list[str] = []

    # Na raiz, coloca o README.md principal primeiro antes das pastas
    if depth == 0:
        for file_page in node["files"]:
            items_html.append(render_file_item_html(file_page))

    # Pastas ordenadas alfabeticamente
    for folder_name in sorted(node["dirs"].keys()):
        child_node = node["dirs"][folder_name]
        children_html = render_tree_html(child_node, depth + 1)
        is_api_dir = child_node["path"] == "backend/docs"
        folder_badge = '<span class="tree-badge api-badge">API</span>' if is_api_dir else ""
        items_html.append(
            f"""
            <li class="tree-folder" data-folder-path="{html.escape(child_node['path'])}">
                <div class="tree-folder-row" onclick="toggleFolder(this)" title="{html.escape(child_node['path'])}/">
                    <span class="tree-chevron">▾</span>
                    <span class="tree-folder-icon">📁</span>
                    <span class="tree-folder-name">{html.escape(folder_name)}/</span>
                    {folder_badge}
                </div>
                <ul class="tree-children">
                    {children_html}
                </ul>
            </li>
            """
        )

    # Arquivos dentro das subpastas
    if depth > 0:
        # Ordena colocando README.md primeiro dentro de cada pasta
        sorted_files = sorted(
            node["files"],
            key=lambda p: (0 if p["name"].lower() == "readme.md" else 1, p["name"].lower()),
        )
        for file_page in sorted_files:
            items_html.append(render_file_item_html(file_page))

    return "\n".join(items_html)


def render_file_item_html(page: dict[str, Any]) -> str:
    """Renderiza um item de arquivo na árvore lateral."""
    kind = page["kind"]
    if kind == "api_iframe":
        icon = "⚡" if "swagger" in page["name"].lower() else "📘"
        badge = '<span class="tree-badge interactive-badge">Interativo</span>'
    elif kind == "openapi_json":
        icon = "📋"
        badge = '<span class="tree-badge json-badge">JSON</span>'
    elif kind == "gallery":
        icon = "🖼️"
        badge = ""
    else:
        icon = "📄"
        badge = ""

    return f"""
    <li class="tree-file" data-page-id="{html.escape(page['id'])}" data-path="{html.escape(page['path'])}" data-title="{html.escape(page['title'])}">
        <a href="#page={html.escape(page['id'])}" class="tree-file-link" onclick="selectPage('{html.escape(page['id'])}', event)">
            <span class="tree-file-icon">{icon}</span>
            <span class="tree-file-labels">
                <span class="tree-file-name">{html.escape(page['name'])}{badge}</span>
                <span class="tree-file-subtitle">{html.escape(page['title'])}</span>
            </span>
        </a>
    </li>
    """


def generate_html() -> None:
    """Executa todo o pipeline de compilação e escreve `docs/index.html`."""
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
            iframe_src = f"api/{api_file_name}"
            pages.append(
                {
                    "id": rel_path,
                    "path": rel_path,
                    "name": api_file_name,
                    "title": title,
                    "kind": "api_iframe",
                    "iframe_src": iframe_src,
                    "html": "",
                    "toc": [],
                    "search_text": f"{title} {rel_path} API REST FastAPI Swagger ReDoc OpenAPI endpoints",
                }
            )
        elif suffix == ".json" and rel_path.startswith("backend/docs/"):
            pages.append(build_openapi_json_page(rel_path, title))

    # Inclui galerias de diagramas de `artefatos/` na árvore
    pages.extend(virtual_diagram_pages)

    tree = build_file_tree(pages)
    tree_html = render_tree_html(tree)

    pygments_light_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    pygments_dark_css = HtmlFormatter(style="monokai").get_style_defs(
        '[data-theme="dark"] .codehilite'
    )

    pages_json = json.dumps(pages, ensure_ascii=False)
    build_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

    html_doc = f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="light">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Hermes — Documentação Oficial & API</title>
    <style>
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #eef1f5;
            --bg-hover: #eaeef2;
            --bg-active: #dbeafe;
            --text-primary: #1f2328;
            --text-secondary: #57606a;
            --text-muted: #6e7781;
            --border-color: #d0d7de;
            --accent-color: #0969da;
            --accent-hover: #0550ae;
            --code-bg: #f6f8fa;
            --blockquote-bg: #f0f7ff;
            --blockquote-border: #0969da;
            --shadow-sm: 0 1px 3px rgba(27, 31, 36, 0.08);
            --header-height: 58px;
            --sidebar-width: 320px;
            --toc-width: 250px;
        }}

        [data-theme="dark"] {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --bg-hover: #1f242c;
            --bg-active: #1f3a5f;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-color: #30363d;
            --accent-color: #58a6ff;
            --accent-hover: #79c0ff;
            --code-bg: #161b22;
            --blockquote-bg: #121d2f;
            --blockquote-border: #58a6ff;
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}

        /* Top Navbar */
        .topbar {{
            height: var(--header-height);
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
            z-index: 20;
        }}

        .topbar-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .menu-toggle {{
            display: none;
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-radius: 6px;
            padding: 5px 9px;
            cursor: pointer;
            font-size: 16px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            color: var(--text-primary);
            font-weight: 700;
            font-size: 18px;
        }}

        .brand-badge {{
            font-size: 11px;
            font-weight: 600;
            background-color: var(--bg-tertiary);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            padding: 2px 8px;
            border-radius: 12px;
        }}

        .topbar-right {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .top-api-btn {{
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .top-api-btn:hover {{
            border-color: var(--accent-color);
            color: var(--accent-color);
        }}

        .theme-btn {{
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }}

        .theme-btn:hover {{
            border-color: var(--accent-color);
        }}

        /* Main Workspace Layout */
        .workspace {{
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }}

        /* Left Sidebar: File Tree */
        .sidebar {{
            width: var(--sidebar-width);
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            user-select: none;
        }}

        .sidebar-search {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}

        .search-input {{
            width: 100%;
            padding: 7px 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-size: 13px;
            outline: none;
        }}

        .search-input:focus {{
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(9, 105, 218, 0.2);
        }}

        .sidebar-toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 14px 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        .tree-actions button {{
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 11px;
            padding: 2px 5px;
            border-radius: 4px;
        }}

        .tree-actions button:hover {{
            background-color: var(--bg-hover);
            color: var(--accent-color);
        }}

        .tree-container {{
            flex: 1;
            overflow-y: auto;
            padding: 6px 8px 20px;
        }}

        .tree-root, .tree-children {{
            list-style: none;
            margin: 0;
            padding: 0;
        }}

        .tree-children {{
            padding-left: 15px;
            border-left: 1px solid var(--border-color);
            margin-left: 10px;
        }}

        .tree-folder.collapsed > .tree-children {{
            display: none;
        }}

        .tree-folder.collapsed > .tree-folder-row .tree-chevron {{
            transform: rotate(-90deg);
        }}

        .tree-folder-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .tree-folder-row:hover {{
            background-color: var(--bg-hover);
        }}

        .tree-chevron {{
            font-size: 11px;
            width: 12px;
            text-align: center;
            color: var(--text-muted);
            transition: transform 0.15s ease;
        }}

        .tree-file-link {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 6px 8px;
            margin: 1px 0;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-primary);
            transition: background-color 0.12s ease;
        }}

        .tree-file-link:hover {{
            background-color: var(--bg-hover);
        }}

        .tree-file.active > .tree-file-link {{
            background-color: var(--bg-active);
            color: var(--accent-color);
            font-weight: 600;
        }}

        .tree-file-icon {{
            font-size: 14px;
            line-height: 1.4;
            flex-shrink: 0;
        }}

        .tree-file-labels {{
            display: flex;
            flex-direction: column;
            min-width: 0;
        }}

        .tree-file-name {{
            font-size: 13px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            word-break: break-all;
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}

        .tree-file-subtitle {{
            font-size: 11.5px;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .tree-badge {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 10px;
            font-weight: 700;
            padding: 1px 6px;
            border-radius: 10px;
            text-transform: uppercase;
        }}

        .api-badge {{
            background-color: #dbeafe;
            color: #1d4ed8;
        }}

        .interactive-badge {{
            background-color: #dcfce7;
            color: #15803d;
        }}

        .json-badge {{
            background-color: #fef3c7;
            color: #b45309;
        }}

        [data-theme="dark"] .api-badge {{
            background-color: #1e3a8a;
            color: #93c5fd;
        }}

        [data-theme="dark"] .interactive-badge {{
            background-color: #14532d;
            color: #86efac;
        }}

        [data-theme="dark"] .json-badge {{
            background-color: #78350f;
            color: #fcd34d;
        }}

        /* Search Results Box */
        .search-results {{
            display: none;
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-color);
            max-height: 260px;
            overflow-y: auto;
            background-color: var(--bg-primary);
        }}

        .search-result-item {{
            display: block;
            padding: 7px 8px;
            border-radius: 6px;
            text-decoration: none;
            color: var(--text-primary);
            font-size: 12.5px;
            border-bottom: 1px solid var(--border-color);
        }}

        .search-result-item:last-child {{
            border-bottom: none;
        }}

        .search-result-item:hover {{
            background-color: var(--bg-hover);
        }}

        .search-result-path {{
            font-family: ui-monospace, SFMono-Regular, monospace;
            font-size: 11px;
            color: var(--accent-color);
        }}

        /* Center Content Area */
        .main-stage {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background-color: var(--bg-primary);
        }}

        .breadcrumb-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 28px;
            background-color: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
            flex-shrink: 0;
            gap: 12px;
        }}

        .breadcrumb-path {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}

        .breadcrumb-segment {{
            color: var(--text-secondary);
        }}

        .breadcrumb-current {{
            color: var(--text-primary);
            font-weight: 600;
        }}

        .breadcrumb-actions {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .content-scroll {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            justify-content: center;
        }}

        .markdown-body {{
            width: 100%;
            max-width: 900px;
            padding: 32px 40px 80px;
        }}

        /* Full-height Iframe for Interactive API pages */
        .iframe-stage {{
            flex: 1;
            width: 100%;
            height: 100%;
            border: none;
            display: none;
            background-color: #ffffff;
        }}

        /* Right Table of Contents */
        .toc-sidebar {{
            width: var(--toc-width);
            border-left: 1px solid var(--border-color);
            background-color: var(--bg-primary);
            padding: 20px 16px;
            overflow-y: auto;
            flex-shrink: 0;
            font-size: 12.5px;
        }}

        .toc-sidebar.hidden {{
            display: none;
        }}

        .toc-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 10px;
        }}

        .toc-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}

        .toc-list li {{
            margin: 4px 0;
        }}

        .toc-list a {{
            color: var(--text-secondary);
            text-decoration: none;
            display: block;
            padding: 3px 6px;
            border-radius: 4px;
            line-height: 1.35;
        }}

        .toc-list a:hover {{
            color: var(--accent-color);
            background-color: var(--bg-hover);
        }}

        .toc-sublist {{
            list-style: none;
            padding-left: 12px;
            margin: 2px 0;
            border-left: 1px solid var(--border-color);
        }}

        /* Markdown Typography & Elements */
        .markdown-body h1,
        .markdown-body h2,
        .markdown-body h3,
        .markdown-body h4 {{
            color: var(--text-primary);
            font-weight: 600;
            line-height: 1.25;
            margin-top: 24px;
            margin-bottom: 16px;
        }}

        .markdown-body h1 {{
            font-size: 2em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid var(--border-color);
            margin-top: 0;
        }}

        .markdown-body h2 {{
            font-size: 1.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid var(--border-color);
        }}

        .markdown-body h3 {{
            font-size: 1.25em;
        }}

        .markdown-body a {{
            color: var(--accent-color);
            text-decoration: none;
        }}

        .markdown-body a:hover {{
            text-decoration: underline;
        }}

        .markdown-body p,
        .markdown-body ul,
        .markdown-body ol,
        .markdown-body blockquote,
        .markdown-body table {{
            margin-top: 0;
            margin-bottom: 16px;
        }}

        .markdown-body code {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 85%;
            background-color: var(--bg-tertiary);
            padding: 0.2em 0.4em;
            border-radius: 6px;
        }}

        .markdown-body pre {{
            background-color: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            position: relative;
            margin-bottom: 16px;
        }}

        .markdown-body pre code {{
            background: none;
            padding: 0;
            font-size: 13px;
            line-height: 1.45;
        }}

        .copy-code-btn {{
            position: absolute;
            top: 8px;
            right: 8px;
            background-color: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 5px;
            cursor: pointer;
            opacity: 0.8;
        }}

        .copy-code-btn:hover {{
            opacity: 1;
            border-color: var(--accent-color);
            color: var(--accent-color);
        }}

        .markdown-body blockquote {{
            margin-left: 0;
            padding: 12px 16px;
            background-color: var(--blockquote-bg);
            border-left: 4px solid var(--blockquote-border);
            border-radius: 0 6px 6px 0;
            color: var(--text-primary);
        }}

        .markdown-body blockquote p:last-child {{
            margin-bottom: 0;
        }}

        .markdown-body table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .markdown-body th,
        .markdown-body td {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            text-align: left;
            font-size: 14px;
        }}

        .markdown-body th {{
            background-color: var(--bg-secondary);
            font-weight: 600;
        }}

        .markdown-body img {{
            max-width: 100%;
            height: auto;
        }}

        /* Buttons & API Components */
        .btn-primary, .btn-secondary, .btn-sm {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-decoration: none !important;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .btn-primary {{
            background-color: var(--accent-color);
            color: #ffffff !important;
            padding: 8px 14px;
            font-size: 13px;
            border: 1px solid transparent;
        }}

        .btn-primary:hover {{
            background-color: var(--accent-hover);
        }}

        .btn-secondary {{
            background-color: var(--bg-secondary);
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            font-size: 13px;
        }}

        .btn-secondary:hover {{
            border-color: var(--accent-color);
            color: var(--accent-color) !important;
        }}

        .btn-sm {{
            background-color: var(--bg-secondary);
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color);
            padding: 4px 10px;
            font-size: 12px;
        }}

        .btn-sm:hover {{
            border-color: var(--accent-color);
            color: var(--accent-color) !important;
        }}

        .api-action-bar {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 16px 0 24px;
        }}

        .method-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, monospace;
        }}

        .method-get {{ background-color: #dbeafe; color: #1d4ed8; }}
        .method-post {{ background-color: #dcfce7; color: #15803d; }}
        .method-put {{ background-color: #fef3c7; color: #b45309; }}
        .method-delete {{ background-color: #fee2e2; color: #b91c1c; }}

        /* Diagram Gallery */
        .diagram-gallery {{
            display: flex;
            flex-direction: column;
            gap: 24px;
            margin-top: 20px;
        }}

        .diagram-card {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background-color: var(--bg-secondary);
        }}

        .diagram-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 16px;
            border-bottom: 1px solid var(--border-color);
            font-family: ui-monospace, SFMono-Regular, monospace;
            font-size: 13px;
            font-weight: 600;
        }}

        .diagram-img-wrap {{
            padding: 16px;
            background-color: #ffffff;
            text-align: center;
            overflow-x: auto;
        }}

        /* Pygments Syntax Highlighting */
        {pygments_light_css}
        {pygments_dark_css}

        /* Responsive Mobile / Tablet */
        @media (max-width: 1100px) {{
            .toc-sidebar {{
                display: none;
            }}
        }}

        @media (max-width: 768px) {{
            .menu-toggle {{
                display: inline-block;
            }}
            .sidebar {{
                position: absolute;
                top: 0;
                left: 0;
                bottom: 0;
                z-index: 30;
                transform: translateX(-100%);
                transition: transform 0.2s ease;
                box-shadow: 2px 0 12px rgba(0, 0, 0, 0.2);
            }}
            .sidebar.open {{
                transform: translateX(0);
            }}
            .markdown-body {{
                padding: 20px 18px 60px;
            }}
            .top-api-btn span.label-text {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <!-- Top Header -->
    <header class="topbar">
        <div class="topbar-left">
            <button class="menu-toggle" onclick="toggleMobileSidebar()" title="Abrir menu">☰</button>
            <a href="#page=README.md" class="brand" onclick="selectPage('README.md', event)">
                <span>📞 Hermes Docs</span>
                <span class="brand-badge">MJSP</span>
            </a>
        </div>
        <div class="topbar-right">
            <a href="#page=backend/docs/swagger.html" class="top-api-btn" onclick="selectPage('backend/docs/swagger.html', event)" title="Documentação Interativa Swagger / Scalar">
                <span>⚡</span>
                <span class="label-text">Swagger API</span>
            </a>
            <a href="#page=backend/docs/openapi.html" class="top-api-btn" onclick="selectPage('backend/docs/openapi.html', event)" title="Documentação Interativa ReDoc">
                <span>📘</span>
                <span class="label-text">OpenAPI (ReDoc)</span>
            </a>
            <button class="theme-btn" id="themeToggleBtn" onclick="toggleTheme()" title="Alternar tema claro/escuro">
                🌙
            </button>
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="workspace">
        <!-- Left File-Tree Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-search">
                <input
                    type="search"
                    id="searchInput"
                    class="search-input"
                    placeholder="Buscar arquivo ou conteúdo..."
                    oninput="handleSearch(this.value)"
                />
            </div>
            <div id="searchResults" class="search-results"></div>
            <div class="sidebar-toolbar">
                <span>Arquivos do Repositório</span>
                <div class="tree-actions">
                    <button onclick="expandAllFolders(true)" title="Expandir tudo">Expandir</button>
                    <button onclick="expandAllFolders(false)" title="Recolher tudo">Recolher</button>
                </div>
            </div>
            <nav class="tree-container">
                <ul class="tree-root">
                    {tree_html}
                </ul>
            </nav>
        </aside>

        <!-- Center Stage -->
        <main class="main-stage">
            <div class="breadcrumb-bar">
                <div class="breadcrumb-path" id="breadcrumbPath">
                    <span>mjsp_hermes</span> / <span class="breadcrumb-current">README.md</span>
                </div>
                <div class="breadcrumb-actions" id="breadcrumbActions"></div>
            </div>

            <div class="content-scroll" id="contentScroll">
                <article class="markdown-body" id="markdownViewer"></article>
            </div>

            <iframe id="apiIframe" class="iframe-stage" title="Documentação Interativa da API"></iframe>
        </main>

        <!-- Right Table of Contents -->
        <aside class="toc-sidebar" id="tocSidebar">
            <div class="toc-title">Nesta Página</div>
            <ul class="toc-list" id="tocList"></ul>
        </aside>
    </div>

    <script>
        const PAGES = {pages_json};
        const PAGE_MAP = Object.fromEntries(PAGES.map(p => [p.id, p]));
        let currentPageId = "README.md";

        function initTheme() {{
            const saved = localStorage.getItem("hermes_docs_theme");
            if (saved === "dark" || saved === "light") {{
                document.documentElement.setAttribute("data-theme", saved);
            }}
            updateThemeButton();
        }}

        function toggleTheme() {{
            const current = document.documentElement.getAttribute("data-theme") || "light";
            const next = current === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", next);
            localStorage.setItem("hermes_docs_theme", next);
            updateThemeButton();
        }}

        function updateThemeButton() {{
            const current = document.documentElement.getAttribute("data-theme") || "light";
            const btn = document.getElementById("themeToggleBtn");
            btn.textContent = current === "dark" ? "☀️" : "🌙";
        }}

        function toggleFolder(rowEl) {{
            const folderLi = rowEl.closest(".tree-folder");
            if (folderLi) {{
                folderLi.classList.toggle("collapsed");
            }}
        }}

        function expandAllFolders(expand) {{
            document.querySelectorAll(".tree-folder").forEach(el => {{
                if (expand) {{
                    el.classList.remove("collapsed");
                }} else {{
                    el.classList.add("collapsed");
                }}
            }});
        }}

        function toggleMobileSidebar() {{
            document.getElementById("sidebar").classList.toggle("open");
        }}

        function updateBreadcrumbs(page) {{
            const container = document.getElementById("breadcrumbPath");
            const actions = document.getElementById("breadcrumbActions");
            const parts = page.path.split("/");
            let htmlParts = ['<span class="breadcrumb-segment">mjsp_hermes</span>'];
            parts.forEach((part, idx) => {{
                if (idx === parts.length - 1) {{
                    htmlParts.push(`<span class="breadcrumb-current">${{part}}</span>`);
                }} else {{
                    htmlParts.push(`<span class="breadcrumb-segment">${{part}}</span>`);
                }}
            }});
            container.innerHTML = htmlParts.join(' <span>/</span> ');

            if (page.kind === "api_iframe") {{
                actions.innerHTML = `
                    <a href="${{page.iframe_src}}" target="_blank" rel="noopener" class="btn-sm">
                        Abrir em nova aba ↗
                    </a>
                    <a href="api/openapi.json" target="_blank" rel="noopener" class="btn-sm" download="openapi.json">
                        openapi.json ⬇
                    </a>
                `;
            }} else {{
                actions.innerHTML = `<span style="font-size:11.5px;color:var(--text-muted)">Atualizado em {build_timestamp}</span>`;
            }}
        }}

        function renderToc(tocTokens) {{
            const tocSidebar = document.getElementById("tocSidebar");
            const tocList = document.getElementById("tocList");

            if (!tocTokens || tocTokens.length === 0) {{
                tocSidebar.classList.add("hidden");
                tocList.innerHTML = "";
                return;
            }}

            tocSidebar.classList.remove("hidden");
            const items = tocTokens.map(token => {{
                let subHtml = "";
                if (token.children && token.children.length > 0) {{
                    const childrenLinks = token.children.map(child =>
                        `<li><a href="javascript:void(0)" onclick="scrollToHeading('${{child.id}}')">${{child.name}}</a></li>`
                    ).join("");
                    subHtml = `<ul class="toc-sublist">${{childrenLinks}}</ul>`;
                }}
                return `<li>
                    <a href="javascript:void(0)" onclick="scrollToHeading('${{token.id}}')">${{token.name}}</a>
                    ${{subHtml}}
                </li>`;
            }}).join("");

            tocList.innerHTML = items;
        }}

        function scrollToHeading(headingId) {{
            const target = document.getElementById(headingId);
            if (target) {{
                target.scrollIntoView({{ behavior: "smooth", block: "start" }});
            }}
        }}

        function attachCodeCopyButtons() {{
            const viewer = document.getElementById("markdownViewer");
            viewer.querySelectorAll("pre").forEach(pre => {{
                if (pre.querySelector(".copy-code-btn")) return;
                const btn = document.createElement("button");
                btn.className = "copy-code-btn";
                btn.textContent = "Copiar";
                btn.onclick = () => {{
                    const code = pre.querySelector("code");
                    const text = code ? code.innerText : pre.innerText;
                    navigator.clipboard.writeText(text).then(() => {{
                        btn.textContent = "Copiado!";
                        setTimeout(() => {{ btn.textContent = "Copiar"; }}, 1500);
                    }});
                }};
                pre.appendChild(btn);
            }});
        }}

        function highlightActiveTreeItem(pageId) {{
            document.querySelectorAll(".tree-file").forEach(el => {{
                if (el.getAttribute("data-page-id") === pageId) {{
                    el.classList.add("active");
                    // Garante que as pastas ancestrais estejam abertas
                    let parent = el.parentElement;
                    while (parent) {{
                        if (parent.classList && parent.classList.contains("tree-folder")) {{
                            parent.classList.remove("collapsed");
                        }}
                        parent = parent.parentElement;
                    }}
                }} else {{
                    el.classList.remove("active");
                }}
            }});
        }}

        function selectPage(pageId, event, sectionId = null) {{
            if (event) {{
                event.preventDefault();
            }}
            const page = PAGE_MAP[pageId] || PAGE_MAP["README.md"];
            if (!page) return;

            currentPageId = page.id;
            highlightActiveTreeItem(page.id);
            updateBreadcrumbs(page);

            const contentScroll = document.getElementById("contentScroll");
            const markdownViewer = document.getElementById("markdownViewer");
            const apiIframe = document.getElementById("apiIframe");
            const tocSidebar = document.getElementById("tocSidebar");

            if (page.kind === "api_iframe") {{
                contentScroll.style.display = "none";
                tocSidebar.classList.add("hidden");
                apiIframe.style.display = "block";
                if (apiIframe.getAttribute("src") !== page.iframe_src) {{
                    apiIframe.setAttribute("src", page.iframe_src);
                }}
            }} else {{
                apiIframe.style.display = "none";
                contentScroll.style.display = "flex";
                markdownViewer.innerHTML = page.html;
                renderToc(page.toc);
                attachCodeCopyButtons();

                if (sectionId) {{
                    setTimeout(() => scrollToHeading(sectionId), 50);
                }} else {{
                    contentScroll.scrollTop = 0;
                }}
            }}

            // Fecha sidebar mobile se aberta
            document.getElementById("sidebar").classList.remove("open");

            // Atualiza hash na URL sem recarregar
            let newHash = `#page=${{page.id}}`;
            if (sectionId) {{
                newHash += `&section=${{encodeURIComponent(sectionId)}}`;
            }}
            if ( decodeURIComponent(window.location.hash) !== decodeURIComponent(newHash) ) {{
                history.pushState(null, "", newHash);
            }}
        }}

        function handleHashChange() {{
            const rawHash = window.location.hash.replace(/^#/, "");
            if (!rawHash) {{
                selectPage("README.md");
                return;
            }}
            const params = new URLSearchParams(rawHash);
            const pageId = params.get("page");
            const sectionId = params.get("section");
            if (pageId && PAGE_MAP[pageId]) {{
                selectPage(pageId, null, sectionId);
            }} else {{
                // Caso seja uma âncora simples dentro da página atual
                scrollToHeading(rawHash);
            }}
        }}

        function handleSearch(query) {{
            const resultsBox = document.getElementById("searchResults");
            const q = query.trim().toLowerCase();
            if (!q) {{
                resultsBox.style.display = "none";
                resultsBox.innerHTML = "";
                return;
            }}

            const matches = PAGES.filter(p =>
                p.path.toLowerCase().includes(q) ||
                p.title.toLowerCase().includes(q) ||
                p.search_text.toLowerCase().includes(q)
            ).slice(0, 12);

            if (matches.length === 0) {{
                resultsBox.style.display = "block";
                resultsBox.innerHTML = '<div style="padding:6px 8px;font-size:12px;color:var(--text-muted)">Nenhum resultado encontrado.</div>';
                return;
            }}

            resultsBox.style.display = "block";
            resultsBox.innerHTML = matches.map(m => `
                <a href="#page=${{m.id}}" class="search-result-item" onclick="selectPage('${{m.id}}', event); document.getElementById('searchResults').style.display='none';">
                    <div style="font-weight:600">${{m.title}}</div>
                    <div class="search-result-path">${{m.path}}</div>
                </a>
            `).join("");
        }}

        window.addEventListener("hashchange", handleHashChange);
        window.addEventListener("DOMContentLoaded", () => {{
            initTheme();
            handleHashChange();
        }});
    </script>
</body>
</html>
"""

    OUTPUT_HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"[OK] Documentação gerada com sucesso em: {OUTPUT_HTML_PATH}")
    print(f"[OK] Total de páginas indexadas na árvore: {len(pages)}")


if __name__ == "__main__":
    generate_html()
