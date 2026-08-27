from transformers import pipeline

models = [
    'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    'infly/inf-retriever-v1-1.5b',
    'voyageai/voyage-4-nano',
    'intfloat/multilingual-e5-large-instruct',
    'sergeyzh/BERTA',
    'ai-forever/FRIDA',
    'sergeyzh/rubert-mini-frida',
    'deepvk/USER2-base',
    'jinaai/jina-embeddings-v5-text-small',
    'Qwen/Qwen3-Embedding-0.6B'
]

#classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
classifier = pipeline("zero-shot-classification", model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
'''sent = "Solicitante: Socorro! Tem um homem me ameaçando com uma faca! Oh não, ele acabou de me dar uma facada"
sent += "Atendente: Aqui é do atendimento de emergencias, vamos enviar uma viatura imediatamente."
sent += "Solicitante: Se vocês não vierem logo EU VOU MORRER!. Está acontecendo agora mesmo!"
sent += "Atendente: Você está sozinha?"
sent += "Solicitante: Sim, a rua está deserta. Sem risco de tumulto."'''

sent = """
Solicitante: Meu marido me bateu ontem. Quero denunciar!
Atendente: Sinto muito que isso tenha acontecido. Você está em segurança agora?
Solicitante: Sim, estou na casa da minha irmã.
Atendente: Entendo. Você gostaria de registrar um boletim de ocorrência?
Solicitante: Sim, por favor. E gostaria de saber se há uma delegacia da mulher por perto.
"""

fatoOcorrendoNesteMomento = "Fato Ocorrendo Neste Momento"
autorDoFatoNoLocal = "Autor Do Fato No Local"
autorDoFatoArmado = "Autor Do Fato Armado"
feridosComRiscoDeMorte = "Feridos Com Risco de Morte"
riscoDeTumulto = "Risco De Tumulto"
leiMariaPenha = "Lei Maria da Penha"

clfnames = [
    fatoOcorrendoNesteMomento,
    autorDoFatoNoLocal,
    autorDoFatoArmado,
    feridosComRiscoDeMorte,
    riscoDeTumulto,
    leiMariaPenha,
]
res = classifier(sent, clfnames, multi_label=True)
res2 = {label: score for score, label in zip(res['scores'], res['labels'])}
print(res2)