const PAGE_MAP = Object.fromEntries(PAGES.map(p => [p.id, p]));
let currentPageId = "README.md";

function initTheme() {
    const saved = localStorage.getItem("hermes_docs_theme");
    if (saved === "dark" || saved === "light") {
        document.documentElement.setAttribute("data-theme", saved);
    }
    updateThemeButton();
}

function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("hermes_docs_theme", next);
    updateThemeButton();
}

function updateThemeButton() {
    const current = document.documentElement.getAttribute("data-theme") || "light";
    const btn = document.getElementById("themeToggleBtn");
    btn.textContent = current === "dark" ? "☀️" : "🌙";
}

function toggleFolder(rowEl) {
    const folderLi = rowEl.closest(".tree-folder");
    if (folderLi) {
        folderLi.classList.toggle("collapsed");
    }
}

function expandAllFolders(expand) {
    document.querySelectorAll(".tree-folder").forEach(el => {
        if (expand) {
            el.classList.remove("collapsed");
        } else {
            el.classList.add("collapsed");
        }
    });
}

function toggleMobileSidebar() {
    document.getElementById("sidebar").classList.toggle("open");
}

function updateBreadcrumbs(page) {
    const container = document.getElementById("breadcrumbPath");
    const actions = document.getElementById("breadcrumbActions");
    const parts = page.path.split("/");
    let htmlParts = ['<span class="breadcrumb-segment">mjsp_hermes</span>'];
    parts.forEach((part, idx) => {
        if (idx === parts.length - 1) {
            htmlParts.push(`<span class="breadcrumb-current">${part}</span>`);
        } else {
            htmlParts.push(`<span class="breadcrumb-segment">${part}</span>`);
        }
    });
    container.innerHTML = htmlParts.join(" <span>/</span> ");

    if (page.kind === "api_iframe") {
        actions.innerHTML = `
            <a href="${page.iframe_src}" target="_blank" rel="noopener" class="btn-sm">
                Abrir em nova aba ↗
            </a>
            <a href="api/openapi.json" target="_blank" rel="noopener" class="btn-sm" download="openapi.json">
                openapi.json ⬇
            </a>
        `;
    } else {
        actions.innerHTML = `<span class="build-timestamp">Atualizado em ${BUILD_TIMESTAMP}</span>`;
    }
}

function renderToc(tocTokens) {
    const tocSidebar = document.getElementById("tocSidebar");
    const tocList = document.getElementById("tocList");

    if (!tocTokens || tocTokens.length === 0) {
        tocSidebar.classList.add("hidden");
        tocList.innerHTML = "";
        return;
    }

    tocSidebar.classList.remove("hidden");
    const items = tocTokens.map(token => {
        let subHtml = "";
        if (token.children && token.children.length > 0) {
            const childrenLinks = token.children.map(child =>
                `<li><a href="javascript:void(0)" onclick="scrollToHeading('${child.id}')">${child.name}</a></li>`
            ).join("");
            subHtml = `<ul class="toc-sublist">${childrenLinks}</ul>`;
        }
        return `<li>
            <a href="javascript:void(0)" onclick="scrollToHeading('${token.id}')">${token.name}</a>
            ${subHtml}
        </li>`;
    }).join("");

    tocList.innerHTML = items;
}

function scrollToHeading(headingId) {
    const target = document.getElementById(headingId);
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function attachCodeCopyButtons() {
    const viewer = document.getElementById("markdownViewer");
    viewer.querySelectorAll("pre").forEach(pre => {
        if (pre.querySelector(".copy-code-btn")) return;
        const btn = document.createElement("button");
        btn.className = "copy-code-btn";
        btn.textContent = "Copiar";
        btn.onclick = () => {
            const code = pre.querySelector("code");
            const text = code ? code.innerText : pre.innerText;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = "Copiado!";
                setTimeout(() => { btn.textContent = "Copiar"; }, 1500);
            });
        };
        pre.appendChild(btn);
    });
}

function highlightActiveTreeItem(pageId) {
    document.querySelectorAll(".tree-file").forEach(el => {
        if (el.getAttribute("data-page-id") === pageId) {
            el.classList.add("active");
            let parent = el.parentElement;
            while (parent) {
                if (parent.classList && parent.classList.contains("tree-folder")) {
                    parent.classList.remove("collapsed");
                }
                parent = parent.parentElement;
            }
        } else {
            el.classList.remove("active");
        }
    });
}

function selectPage(pageId, event, sectionId = null) {
    if (event) {
        event.preventDefault();
    }
    const page = PAGE_MAP[pageId] || PAGE_MAP["README.md"];
    if (!page) return;

    currentPageId = page.id;
    highlightActiveTreeItem(page.id);
    updateBreadcrumbs(page);

    const contentScroll = document.getElementById("contentScroll");
    const markdownViewer = document.getElementById("markdownViewer");
    const apiIframe = document.getElementById("apiIframe");
    const tocSidebar = document.getElementById("tocSidebar");

    if (page.kind === "api_iframe") {
        contentScroll.style.display = "none";
        tocSidebar.classList.add("hidden");
        apiIframe.style.display = "block";
        if (apiIframe.getAttribute("src") !== page.iframe_src) {
            apiIframe.setAttribute("src", page.iframe_src);
        }
    } else {
        apiIframe.style.display = "none";
        contentScroll.style.display = "flex";
        markdownViewer.innerHTML = page.html;
        renderToc(page.toc);
        attachCodeCopyButtons();

        if (sectionId) {
            setTimeout(() => scrollToHeading(sectionId), 50);
        } else {
            contentScroll.scrollTop = 0;
        }
    }

    document.getElementById("sidebar").classList.remove("open");

    let newHash = `#page=${page.id}`;
    if (sectionId) {
        newHash += `&section=${encodeURIComponent(sectionId)}`;
    }
    if (decodeURIComponent(window.location.hash) !== decodeURIComponent(newHash)) {
        history.pushState(null, "", newHash);
    }
}

function handleHashChange() {
    const rawHash = window.location.hash.replace(/^#/, "");
    if (!rawHash) {
        selectPage("README.md");
        return;
    }
    const params = new URLSearchParams(rawHash);
    const pageId = params.get("page");
    const sectionId = params.get("section");
    if (pageId && PAGE_MAP[pageId]) {
        selectPage(pageId, null, sectionId);
    } else {
        scrollToHeading(rawHash);
    }
}

function handleSearch(query) {
    const resultsBox = document.getElementById("searchResults");
    const q = query.trim().toLowerCase();
    if (!q) {
        resultsBox.style.display = "none";
        resultsBox.innerHTML = "";
        return;
    }

    const matches = PAGES.filter(p =>
        p.path.toLowerCase().includes(q) ||
        p.title.toLowerCase().includes(q) ||
        p.search_text.toLowerCase().includes(q)
    ).slice(0, 12);

    if (matches.length === 0) {
        resultsBox.style.display = "block";
        resultsBox.innerHTML = '<div class="search-empty">Nenhum resultado encontrado.</div>';
        return;
    }

    resultsBox.style.display = "block";
    resultsBox.innerHTML = matches.map(m => `
        <a href="#page=${m.id}" class="search-result-item" onclick="selectPage('${m.id}', event); document.getElementById('searchResults').style.display='none';">
            <div style="font-weight:600">${m.title}</div>
            <div class="search-result-path">${m.path}</div>
        </a>
    `).join("");
}

window.addEventListener("hashchange", handleHashChange);
window.addEventListener("DOMContentLoaded", () => {
    initTheme();
    handleHashChange();
});

