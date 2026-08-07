let id_emergencia; 
let atualizarInferencia = false;
let isTranscriptionActive = false;
let recognition;
let transcriptAccumulated = ''; 
let ultimaTranscricaoManual = "";
let inferenceResults = {};
let debounceTimer;
let regexInterval;
const proxy = '/proxy'
const operator_name = 'djangofrontend'

// TODO: Ajustar o flow para fluxo 1 setando assim que inicializar o valor no storage
// TODO: Adicionar flow transcrição navegador e inferencia IA
//TODO: Page Recentes ajustar caso 5XX/4XX retornar page com info sem chamadas recentes


document.addEventListener('DOMContentLoaded', () => {
    const endCallButton = document.getElementById('end-call-button');
    const cancelCallButton = document.getElementById('cancel');
    const transcriptionTextarea = document.getElementById('transcription-text');
    const registerOccurrenceButton = document.getElementById('register_occurrence_button');

    const descriptionModal = document.getElementById('description-modal');

    const name = document.getElementById('name');
    const phone = document.getElementById('phone');
    const address = document.getElementById('address');
    const district = document.getElementById('district');
    const reference = document.getElementById('ref');
    const city = document.getElementById('city');
    const state = document.getElementById('uf');
    const narration = document.getElementById('narracao');
    const resumenInput = document.getElementById('resumen-input');
    const obsInput = document.getElementById('obs-input');
    const resumen = document.getElementById('resumen');
    const obs = document.getElementById('obs');
    const initialNature = document.getElementById('initial-nature');
    const factActive = document.getElementById('fact-active');
    const actorPresent = document.getElementById('actor-present');
    const actorArmed = document.getElementById('actor-armed');
    const leiMariaPenha = document.getElementById('lei-maria-penha');
    const riskOfRiot = document.getElementById('risk-of-riot');
    const riskOfDeath = document.getElementById('risk-of-death');

    function showToast(title, description) {
        alert(title + ': ' + description);
    }

    function clearTranscription(){
        transcriptionTextarea.value = "";
    }

    //Fluxo ativado na configuração
    const selectedFlow = Number(localStorage.getItem("selectedFlow"));
    endCallButton.disabled = false;
    
    endCallButton.disabled = false;
    
    if (selectedFlow === "2" || selectedFlow === "4"){
        resumenInput.classList.add("hidden");
        obsInput.classList.add("hidden");
    } else {
        resumenInput.classList.remove("hidden");
        obsInput.classList.remove("hidden");
    }

    //Atualizando Mapa
    const lat = sessionStorage.getItem('config_lat');
    const lon = sessionStorage.getItem('config_lon');

    if (typeof atualizarMapa === "function") {
        atualizarMapa(lat, lon);
    } else {
        console.warn("Função atualizar Mapa não encontrada.");
    }


    // Id Chamada a ser inferida
    const idSalvo = sessionStorage.getItem("id_emergencia_atual");
    id_emergencia = idSalvo;

    if (idSalvo && selectedFlow === 1) {
        atualizarInferencia = true;
        console.log("Iniciando atendimento automático para ID:", id_emergencia);
        iniciarFluxo(1); 
        
    } else {
        iniciarFluxo(selectedFlow)
    }

    function iniciarFluxo(selectedFlow) {
        switch (selectedFlow) {

            case 1:
                console.log("Fluxo 1: Full Hermes Agents");
                obterInferencia();
                break;

            case 2:
                console.log("Fluxo 2: Transcrição Automática no Navegador e extração com Regex");
                startRecognitionTranscription();
                break;

            case 3:
                console.log("Fluxo 3: Transcrição Manual e Inferência Hermes Agents");
                handleStartManuallyTranscription();
                if (regexInterval) clearInterval(regexInterval);
                regexInterval = setInterval(() => {
                    if (isTranscriptionActive) {
                        const transcription = transcriptionTextarea.value;
                        enviarTranscricaoInterpretador(transcription);
                    }
                }, 2000);
                break;

            case 4:
                console.log("Fluxo 100% Fallback: Transcrição manual e extração com Regex");
                handleStartManuallyTranscription();
                if (regexInterval) clearInterval(regexInterval);
                    regexInterval = setInterval(() => {
                        if (isTranscriptionActive) {
                            const transcription = transcriptionTextarea.value;
                            extrairDadosPorRegex(transcription);
                        }
                    }, 3000);
                break;

            default:
                console.log("Fluxo inválido:", selectedFlow);
                showToast("Erro", "Nenhum fluxo selecionado.");
                break;
        }
    }


    function finalizarFluxo(selectedFlow) {
        switch (selectedFlow) {

            case 1:
                console.log("Finalizando fluxo 1...");
                endCall();
                sendEndSignalToBackend();
                break;

            case 2:
                console.log("Finalizando fluxo 2...");
                endRecognition();
                endCall();
                break;

            case 3:
                console.log("Finalizando fluxo 3...");
                endCall();
                sendEndSignalToBackend();
                break;

            case 4:
                console.log("Finalizando fluxo 4...");
                if (regexInterval) clearInterval(regexInterval);
                const transcription = transcriptionTextarea.value;
                extrairDadosPorRegex(transcription);
                endCall();
                break;

            default:
                console.log("Fluxo inválido:", selectedFlow);
                break;
        }
    }

    function handleStartManuallyTranscription() {
        if (isTranscriptionActive) return;

        limparFormulario();
        isTranscriptionActive = true;
        descriptionModal.textContent = "Realize a transcrição manualmente"

        ultimaTranscricaoManual = "";
    }

    transcriptionTextarea.addEventListener("input", function () {
        const textoAtual = transcriptionTextarea.value;
        let novoTrecho = textoAtual.replace(ultimaTranscricaoManual, "");

        if (!isTranscriptionActive || selectedFlow !== 3) return;

        if (novoTrecho.trim() === "" || textoAtual.length < ultimaTranscricaoManual.length) {
            ultimaTranscricaoManual = textoAtual;
            return;
        }

        if (novoTrecho.endsWith(" ") || novoTrecho.endsWith(".") || novoTrecho.endsWith("\n")) {
            clearTimeout(debounceTimer);
            enviarTranscricaoInterpretador(novoTrecho);
            ultimaTranscricaoManual = textoAtual;
            return;
        }

        clearTimeout(debounceTimer);

        debounceTimer = setTimeout(() => {
            console.log("Enviando bloco de texto manual:", novoTrecho);
            
            enviarTranscricaoInterpretador(novoTrecho);
            ultimaTranscricaoManual = textoAtual;
        }, 1200);
       
    });

    function startRecognitionTranscription() {
        if (recognition) {
            recognition.onend = null; // Remove o auto-restart da instância antiga
            recognition.onerror = null;
            try { recognition.stop(); } catch(e) {}
            recognition = null;
        }
        
        limparFormulario();
        
        if (isTranscriptionActive) return;
        transcriptAccumulated = ''; 
        isTranscriptionActive = true;
        descriptionModal.textContent = "A conversa será transcrita automaticamente."
        
        // Inicializa SpeechRecognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            showToast('Erro', 'Seu navegador não suporta reconhecimento de voz.');
            return;
        }

        recognition = new SpeechRecognition();
        recognition.lang = 'pt-BR';
        recognition.interimResults = true; // resultados parciais
        recognition.continuous = true; // transcrição contínua

        recognition.start();

        recognition.onresult = (event) => {
            let transcriptIncrement = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                
                if (event.results[i].isFinal) {
                    transcriptAccumulated += transcript + '.\n'; 
                     if (selectedFlow === 1) {
                        enviarTranscricaoInterpretador(transcript);
                    }

                    if (selectedFlow === 2) {
                        extrairDadosPorRegex(transcriptAccumulated);
                    }

                } else {
                    transcriptIncrement += transcript;
                }
            }
            
            transcriptionTextarea.value = transcriptAccumulated + transcriptIncrement;
        };

        recognition.onerror = function(event) {
            console.warn('Erro no Reconhecimento de Fala:', event.error);

            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                endRecognition();
                isTranscriptionActive = false;
                showToast('Erro de Microfone', 'Verifique a permissão do microfone no seu navegador.');
                return;
            }

            if (event.error === 'aborted') {
                console.log('Reconhecimento abortado pelo navegador.');
                return;
            }

            if (event.error === 'network') {
                console.log('Erro de rede no reconhecimento.');
            }
        };

        recognition.onend = function() {
            if (!isTranscriptionActive) return;
            console.log("Reiniciando o serviço de reconhecimento de fala...");
            recognition.start();
        };
    }

    function endRecognition(){
        if (recognition) {
            recognition.onend = null;
            recognition.stop();
        }
        isTranscriptionActive = false;
    }
    
    function preencherCamposComInferencias(data) {
        if (!data) return;

        const getFirstValue = (campo) => {
            return Array.isArray(data[campo]) && data[campo].length > 0
                ? data[campo][0].value
                : "";
        };

        name.value = getFirstValue("nome_do_solicitante");
        address.value = getFirstValue("rua_ou_logradouro") + ", " + getFirstValue("numero");
        district.value = getFirstValue("bairro");
        city.value = getFirstValue("municipio");
        reference.value = getFirstValue("ponto_de_referencia");
        initialNature.value = getFirstValue("natureza_inicial");
        narration.value = data["transcricao"] || "";
        transcriptionTextarea.value = data["transcricao"] || "";
        resumen.value = getFirstValue("descricao_breve");
        obs.value = getFirstValue("outras_observacoes");
       
        //NOVOS CAMPOS = True seria o mais prox de 1 (Confirmar com Pitagoras)?
        factActive.value = parseFloat(getFirstValue("Fato Ocorrendo Neste Momento ?")) == 1 ? "sim" : "nao";
        actorPresent.value = parseFloat(getFirstValue("Autor Do Fato No Local ?")) == 1 ? "sim" : "nao";
        actorArmed.value = parseFloat(getFirstValue("Autor Do Fato Armado ?")) == 1 ? "sim" : "nao";
        leiMariaPenha.value = parseFloat(getFirstValue("Lei Maria da Penha ?")) == 1 ? "sim" : "nao";
        riskOfRiot.value = parseFloat(getFirstValue("Risco De Tumulto ?")) == 1 ? "sim" : "nao";
        riskOfDeath.value = parseFloat(getFirstValue("Feridos Com Risco de Morte ?")) == 1 ? "sim" : "nao";

        const updateRiskBadge = (id, value) => {
            const input = document.getElementById(id);
            const statusDot = document.getElementById(id + '-status');
            const container = document.getElementById(id + '-container');

            if (!input || !statusDot) return;

            input.value = value;

            if (value.toLowerCase() === "sim") {
                // Afirmativo
                statusDot.className = "w-3 h-3 rounded-full bg-green-700 mr-2";
                container.className = "flex items-center p-2 border rounded bg-green-20 border-green-100";
            } else if (value.toLowerCase() === "nao") {
                // Negativo
                statusDot.className = "w-3 h-3 rounded-full bg-red-40 mr-2";
                container.className = "flex items-center p-2 border rounded border-red-100 shadow-sm";
            } else {
                // Indeterminado
                statusDot.className = "w-3 h-3 rounded-full bg-gray-300 mr-2";
                container.className = "flex items-center p-2 border rounded bg-gray-50 border-gray-200";
            };
        }
        updateRiskBadge("fact-active", factActive.value);
        updateRiskBadge("actor-present", actorPresent.value);
        updateRiskBadge("actor-armed", actorArmed.value);
        updateRiskBadge("lei-maria-penha", leiMariaPenha.value);
        updateRiskBadge("risk-of-riot", riskOfRiot.value);
        updateRiskBadge("risk-of-death", riskOfDeath.value);
    }

    function extrairDadosPorRegex(textoBruto) {
        const texto = textoBruto
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLowerCase();


        if (texto.includes("chamas") || texto.includes("fogo")) {
            initialNature.value = "Incendio";
        }


        if (texto.includes("chamas") || texto.includes("fogo")) {
            initialNature.value = "Incendio";
        }

        const dados = {};
        const nomeRegex = /\b(?:qual (?:o )?seu nome|falo por gentileza|é|do senhor|da senhora|nome do senhor|nome da senhora|nome por gentileza|qual seu nome|me chamo|eu sou|meu nome é)\s+([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+)?)/i;
        const nomeMatch = texto.match(nomeRegex);
        if (nomeMatch) dados.nome = capitalizar(nomeMatch[1].trim().replace(/[.,]/g, ""));

        const telefoneRegex = /\(?\b\d{2}\)?[\s-]?\d{4,5}[-]?\d{4}\b/;
        const telefoneMatch = texto.match(telefoneRegex);
        if (telefoneMatch) dados.telefone = telefoneMatch[0].replace(/\D/g, '');

        const enderecoRegex = /\b(?:qual endereço|rua|avenida|endereço|que local|km|quilômetro|rodovia|fica na|SQS|QS|QNO|QNE|QSA|Setor|Chacara)\s+([a-zà-ÿ0-9 .-]+?(?=\s(?:bairro|ponto|próximo|$)))/i;
        const enderecoMatch = texto.match(enderecoRegex);
        if (enderecoMatch) dados.endereco = capitalizar(enderecoMatch[1].trim());

        const bairroRegex = /\bbairro\s+([a-zà-ÿ' ]+)/i;
        const bairroMatch = texto.match(bairroRegex);
        if (bairroMatch) dados.bairro = capitalizar(bairroMatch[1].trim());

        const cidadeRegex = /\bcidade\s+([a-zà-ÿ' ]+)/i;
        const cidadeMatch = texto.match(cidadeRegex);
        if (cidadeMatch) dados.cidade = capitalizar(cidadeMatch[1].trim());

        const pontoReferenciaRegex = /\b(?:como chama a loja|lado de que que o senhor falou|ponto de referência ai|ponto de referência|frente ao|lado de|perto de|lado da|lado do|atras de|perto do|dentro do|sentido|proximo ao|perto da)\s+([a-zà-ÿ0-9\s]{1,30})/iu;
        const pontoReferenciaMatch = texto.match(pontoReferenciaRegex);
        if (pontoReferenciaMatch) dados.pontoReferencia = capitalizar(pontoReferenciaMatch[1].trim());

        const ocorrenciaRegex = /\b(ferimento por arma branca|atropelamento|desmaio|convulsao|trauma|queda de arvore|queda da propria altura|acidente|queda|assalto|emergencia medica|morte|sufocado|trauma|isolamento de perímetro|incendio|incendio florestal|captura de insetos|surto|vazamento de gás|engasgamento|assalto|furto|roubo)\b/i;
        const ocorrenciaMatch = texto.match(ocorrenciaRegex);
        if (ocorrenciaMatch) dados.ocorrencia = capitalizar(ocorrenciaMatch[1]);

        dados.nome ? name.value = dados.nome : "";
        dados.telefone ? phone.value = dados.telefone : "";
        dados.endereco ? address.value = dados.endereco : "";
        dados.bairro ? district.value = dados.bairro : "";
        dados.pontoReferencia ? reference.value = dados.pontoReferencia : "";
        dados.cidade ? city.value = dados.cidade : "";
        dados.estado ? state.value = dados.estado : "";
        dados.ocorrencia ? initialNature.value = dados.ocorrencia : "";
        narration.value = texto;

        return dados;
    }

    function enviarTranscricaoInterpretador(textoBruto) {
        if (!id_emergencia) {
            console.error('Não há id_emergencia definido. Abortando envio.');
            return;
        }
        if (!textoBruto || !textoBruto.trim()) {
            console.error('Transcrição vazia. Abortando envio.');
            return;
        }

        console.log('Enviando transcrição. id_emergencia=', id_emergencia);

        const params = new URLSearchParams({
            id_emergencia: String(id_emergencia),
            transcript: textoBruto
        });

        // Envia como POST mas com parâmetros na query (equivalente ao curl -G)
        fetch(`${proxy}/send_transcript/?${params.toString()}`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
        .then(async (response) => {
            if (!response.ok) {
                const errorData = await response.text();
                console.error('Erro no servidor:', response.status, errorData);
                return;
            }
            obterInferencia(); 
            let data;
            try {
                data = await response.json();
                console.log('Transcrição enviada com sucesso:', data);
            } catch (err) {
                console.log('Transcrição enviada (resposta não-JSON)', err);
            }
        })
        .catch(error => {
            console.error('Erro ao enviar transcrição:', error);
        });
    }

    function obterInferencia() {

        if (!atualizarInferencia) return;

        if (atualizarInferencia) {
            if (!id_emergencia) {
                console.error('Não há id_emergencia definido. Abortando obtenção de resultados de inferência.');
            }else{
                fetch(`${proxy}/get_all_inference_results/?id_emergencia=${id_emergencia}`, {
                    method: 'GET',
                    headers: {
                        'Accept': 'application/json'
                    }
                })
                .then(async (response) => {
                    const contentType = response.headers.get('content-type') || '';
                    let data;
                    try {
                        if (contentType.includes('application/json')) {
                            data = await response.json();
                        } else {
                            data = await response.text();
                        }
                    } catch (err) {
                        console.error('Erro ao parsear resposta do servidor:', err);
                        throw err;
                    }

                    if (!response.ok) {
                        console.error('Servidor retornou erro ao obter resultados de inferência. status=', response.status, 'body=', data);
                        return;
                    }

                    console.log('Resultados de inferência obtidos com sucesso. resposta:', data);
                    inferenceResults = data;

                    try {
                        preencherCamposComInferencias(data);

                        // Geocoding em tempo real
                        if (data.lat && data.lon) {
                            const geoEvent = new CustomEvent('novaLocalizacao', { 
                                detail: { lat: data.lat, lon: data.lon } 
                            });
                            window.dispatchEvent(geoEvent);
                        }
                    } catch (e) {
                        console.warn("Não foi possível preencher automaticamente:", e);
                    }
                })
                .catch(error => {
                    console.error('Erro ao obter resultados de inferência:', error);
                });
            }
            setTimeout(obterInferencia, 2000);
        }

    }

        // Function to get CSRF token from cookie
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    function alertEndCall(){
        showToast("Chamada finalizada","ID:" + id_emergencia)
        endCallButton.disabled = true;
    }

    function alertEndCall(){
        showToast("Chamada finalizada","ID:" + id_emergencia)
        endCallButton.disabled = true;
    }

    function sendEndSignalToBackend() {
        atualizarInferencia = false;
        if (!id_emergencia) {
            console.error('Não há id_emergencia definido. Abortando envio do sinal de fim.');
            return;
        }

        const param = new URLSearchParams({
            id_emergencia: String(id_emergencia),
        });

        const param = new URLSearchParams({
            id_emergencia: String(id_emergencia),
        });

        console.log('Enviando sinal de fim de transcrição. id_emergencia=', id_emergencia);
        fetch(`${proxy}/end_call/?${param.toString()}`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            }
        })
        .then(async (response) => {
            const contentType = response.headers.get('content-type') || '';
            let data;
            try {
                if (contentType.includes('application/json')) {
                    data = await response.json();
                } else {
                    data = await response.text();
                }
            } catch (err) {
                console.error('Erro ao parsear resposta do servidor:', err);
                throw err;
            }

            if (!response.ok) {
                console.error('Servidor retornou erro ao finalizar chamada. status=', response.status, 'body=', data);
                return;
            }

            console.log('Sinal de fim de chamada enviado com sucesso. resposta:', data);
            alertEndCall();
        })
        .catch(error => {
            console.error('Erro ao enviar sinal de fim de chamada:', error);
        });
    }

    function cancelCall() {
        isTranscriptionActive = false;
        atualizarInferencia = false;
        transcriptAccumulated = "";
        clearTranscription();
        limparFormulario();

        if (regexInterval) clearInterval(regexInterval);

        if (recognition) {
            recognition.onend = null;
            recognition.stop();
        }
        console.log("Atendimento cancelado e formulário limpo.");
        showToast('Status do Atendimento', 'Cancelado');
    }

    function endCall(){
        isTranscriptionActive = false;
        atualizarInferencia = false;
        alertEndCall();
    }
    
    endCallButton.addEventListener('click', (event) => {
        event.preventDefault();
        finalizarFluxo(selectedFlow)
    });
    cancelCallButton.addEventListener('click', () => cancelCall());
    registerOccurrenceButton.addEventListener('click', () => sendEndSignalToBackend());


    function capitalizar(str) {
        return str.replace(/\b\w/g, c => c.toUpperCase());
    }

    // Máscara telefone
    if (phone) {
        phone.addEventListener('input', (event) => {
            let value = event.target.value.replace(/\D/g, '');
            value = value.replace(/^(\d{2})(\d)/g, "($1) $2");
            value = value.replace(/(\d)(\d{4})$/, "$1-$2");
            event.target.value = value;
        });
    }

    function limparFormulario() {
        const form = document.querySelector("form");
        if (!form) return;
        form.reset();
        const camposTexto = form.querySelectorAll("input[type='text'], textarea");
        camposTexto.forEach(c => c.value = "");
    }
});
