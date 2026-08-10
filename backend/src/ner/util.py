'''
Fake call example:
{
    "Emergencia": {
        "Endereco": {
            "descricao": "Rua Jura Núbia, 223",
            "desc_tipo": "rua_numero",
            "rua": "Rua Jura Núbia",
            "numero": "223",
            "bairro": "",
            "cidade": "Cândido Mendes",
            "estado": "Maranhão",
            "CEP": "65280-000",
            "coords": [
                -1.448297,
                -45.7296635
            ],
            "ref_name": "Cantinho Frango",
            "ref_endereco_completo": "Rua Castelo Branco - Cândido Mendes",
            "ref_tipos": [
                "restaurant",
                "point_of_interest",
                "food",
                "establishment"
            ],
            "ref_distancia": 51.607513556710686
        },
        "Natureza": {
            "Prioridade": "Alerta Vermelho",
            "Natureza": "Incêndio Em Edificação",
            "TiposAgencia": "GM;CdBM;PM",
            "Descrição": "Incêndio em um edifício residencial"
        },
        "Duração da Ligacao (Minutos)": 1.77,
        "Hora": "4h2min"
    },
    "Perfil do Solicitante": {
        "Nome Solicitante": "Mateus Viana",
        "Idade": 89,
        "Genero": "H",
        "Numero": "92593 8190",
        "Instrucao": "Analfabetismo Completo",
        "Envolvimento": "Vitima",
        "Nível de Desespero/Estresse/Medo (0 a 10)": 3
    },
    "roteiro": "Operador: Bom dia, estou aqui para ajudar. O que está acontecendo?\n\nSolicitante: Fogo... no prédio...\n\nOperador: Onde está o incêndio?\n\nSolicitante: No prédio... Jura Núbia... 223.\n\nOperador: Jura Núbia, 223. Tem certeza que é um incêndio?\n\nSolicitante: Sim! Tem fogo! Cheguei pra ver e tinha fumaça, cheirava muito mal.\n\nOperador: Entendo. Você está no prédio?\n\nSolicitante: Sim! Estou aqui. O fogo está dentro.\n\nOperador: Você está seguro?\n\nSolicitante: Estou aqui, tentando sair... É muito quente...\n\nOperador: Tente se afastar das chamas, se possível. Você consegue ver alguma saída?\n\nSolicitante: Não vejo... muita fumaça... estou com dificuldade para respirar...\n\nOperador: Calma, senhor. Tente manter a calma. Você consegue me dizer algo mais sobre o fogo? Tem pessoas dentro?\n\nSolicitante: Não sei... não consigo ver... só vejo fumaça... e fogo...\n\nOperador: Entendi. Pode, por favor, fornecer o seu nome?\n\nSolicitante: Mateus... Mateus Viana.\n\nOperador: Obrigado, Mateus. A sua localização é a Rua Jura Núbia, 223. Fui bem?\n\nSolicitante: Sim, estou aqui.\n\nOperador: Vou acionar a resposta para o incêndio. A equipe de emergência já está a caminho. Fique onde está até que eles cheguem.\n\nSolicitante: Ok... ok...\n"
}
'''

def multi_preds_to_list(preds):
    if isinstance(preds, list):
        return [x for x,_ in preds]
    elif isinstance(preds, str):
        return [preds]
    elif isinstance(preds, dict):
        return [preds[k] for k in preds]
    else:
        raise ValueError(f"Unsupported type for preds: {type(preds)}. Expected list, str, or dict.")