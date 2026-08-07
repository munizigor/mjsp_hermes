const modal = document.getElementById('config-modal');
const openBtn = document.getElementById("open-config-modal");
const closeBtn = document.getElementById("close-config-modal");
const saveConfig = document.getElementById("save-config");
const modoSelect = document.getElementById("modo-operacao");
const inputLat = document.getElementById('lat');
const inputLon = document.getElementById('lon');
const btnStartCall = document.getElementById('start-call-button');
const qtdCallWaiting = document.getElementById('count-waiting');
const qtdCallProgress = document.getElementById('count-active');
const qtdCallFinished = document.getElementById('count-finished');
const proxy = '/proxy'
let qtdCallCompleted = 0;
let qtdCallInProgress = 0;

function atenderChamada(btn) {
    const linha = btn.closest('tr');
    const idExtraido = linha.querySelector('.call-id').innerText.trim();

    console.log("ID Capturado via DOM:", idExtraido);

    sessionStorage.setItem("id_emergencia_atual", idExtraido);

    window.location.href = "/home/";
}

async function carregarChamadas() {
    const tbodyInProgress = document.getElementById('calls-table-body');
    const tbodyFinished = document.getElementById('finished-calls-table-body');

    try {
        const response = await fetch(`${proxy}/list_emergencies/`);
        const dataListCall = await response.json();
        const callInProgress = dataListCall.calls.filter(chamada => chamada.end_time === null);
        const callCompleted = dataListCall.calls.filter(chamada => chamada.end_time !== null);
        qtdCallCompleted = callCompleted.length;
        qtdCallInProgress = callInProgress.length;

        qtdCallProgress.innerText = qtdCallInProgress;
        qtdCallFinished.innerText = qtdCallCompleted

        tbodyInProgress.innerHTML = '';
        tbodyFinished.innerHTML = '';

        if (qtdCallInProgress > 0) {
           callInProgress.forEach(chamada => {
                const tr = document.createElement('tr');
                tr.className = "hover:bg-gray-700/30 transition border-b border-gray-700";
                
                tr.innerHTML = `
                    <td class="px-6 py-4">
                        <div class="text-xs text-gray-500 call-id">${chamada.id}</div>
                    </td>
                    <td class="px-6 py-4 font-mono text-blue-400 text-sm">
                        ${chamada.source_phone_number || '501'}
                    </td>
                    <td class="px-6 py-4 font-mono text-yellow-500">
                        ${chamada.start_time || '2026-03-03 15:02:47'}
                    </td>
                    <td class="px-6 py-4 text-center">
                        <button onclick="atenderChamada(this)" 
                                class="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm font-bold transition">
                            ATENDER
                        </button>
                    </td>
                `;
                tbodyInProgress.appendChild(tr);                
            });
        } else {
            tbodyInProgress.innerHTML = '<tr><td colspan="4" class="p-10 text-center text-gray-500 italic">Aguardando Novas Chamadas</td></tr>';
        }

        if (qtdCallCompleted > 0) {
           callCompleted.forEach(chamada => {
                const tr = document.createElement('tr');
                tr.className = "hover:bg-gray-50";
                
                tr.innerHTML = `
                    <td class="px-3 py-2 text-gray-400 text-xs">
                        <div class="text-xs text-gray-500 call-id">${chamada.id}</div>
                    </td>
                    <td class="px-3 py-2 text-gray-600 font-medium">
                        ${chamada.natureza_inicial || 'Outros'}
                    </td>
                    <td class="px-3 py-2 text-right">
                        <span class="text-[10px] bg-green-100 text-green-700 px-2 py-0.5 rounded-full uppercase">${chamada.status || 'Finalizada'}</span>
                    </td>
                `;
                tbodyFinished.appendChild(tr);
            });
        } else {
            tbodyFinished.innerHTML = '<tr><td colspan="4" class="p-10 text-center text-gray-500 italic">Atualizando...</td></tr>';
        }

    } catch (error) {
        console.error("Erro ao carregar dashboard:", error);
        tbodyInProgress.innerHTML = '<tr><td colspan="4" class="p-10 text-center text-red-400">Erro ao conectar com o servidor.</td></tr>';
        tbodyFinished.innerHTML = '<tr><td colspan="4" class="p-10 text-center text-red-400">Erro ao conectar com o servidor.</td></tr>';
    }
}

carregarChamadas();
setInterval(carregarChamadas, 10000);
    
document.addEventListener('DOMContentLoaded', () => {
    //Configurações
    openBtn.addEventListener("click", () => {
        modal.classList.remove("hidden");
    });

    closeBtn.addEventListener("click", () => {
        modal.classList.add("hidden");
    });

    let selectedFlow = 2;
    const flow = localStorage.getItem("selectedFlow");

    if (flow !== null) {
        document.getElementById("modo-operacao").value = Number(flow);
        console.log("Config carregada do localStorage:", selectedFlow);
    } else {
        console.log("Nenhuma configuração salva. Usando default.");
    }

    function salvarConfiguracao() {

        selectedFlow = Number(modoSelect.value); 

        let lat = -15.7982;
        let lon = -47.8755;

        if (inputLat.value && inputLon.value) {
            const parsedLat = parseFloat(inputLat.value);
            const parsedLon = parseFloat(inputLon.value);
            
            if (parsedLat && parsedLon) {
                lat = parsedLat;
                lon = parsedLon;
                console.log("Coordenadas customizadas detectadas:", lat, lon);
            }
        }
        
        sessionStorage.setItem('config_lat', lat);
        sessionStorage.setItem('config_lon', lon);
        localStorage.setItem("selectedFlow", selectedFlow);

        console.log("Configuração salva. Modo:", selectedFlow, "Localização:", lat, lon);
        modal.classList.add("hidden");
    }

    saveConfig.addEventListener("click", (event) => {
        event.preventDefault();
        salvarConfiguracao();
    });



});

//TODO: Se tbodyInProgress isEmpty, habilitar botão iniciar atendimento