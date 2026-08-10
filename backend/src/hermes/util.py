import json
import sys


sinonimos_ner = {
    "rua": "rua_ou_logradouro",
    "street_number": "numero",
    "number": "numero",
    "cidade": "municipio",
    "endereço_complemento": "complemento",
}


def summarize_context(inferences):
    n_inferences = len(inferences)

    context = {}
    envolvimentos = {}
    # print('Parsing inferences', file=sys.stderr)
    for n, data in enumerate(inferences):
        # print(n, data, file=sys.stderr)
        if type(data["resultado"]) == str:
            inf_dict = json.loads(data["resultado"])
        elif type(data["resultado"]) == dict:
            inf_dict = data["resultado"]
        else:
            inf_dict = {}
        # print('Parsed:', inf_dict, file=sys.stderr)
        place = n / n_inferences * 100
        if "envolvimentos" in inf_dict:
            for e in inf_dict["envolvimentos"]:
                if e["nome_pessoa"] not in envolvimentos:
                    envolvimentos[e["nome_pessoa"]] = {}
                raw_prob = e["probabilidade"]
                if raw_prob > 1:
                    raw_prob = 1.0
                prob_corrected = raw_prob * (place / 100)
                envolvimentos[e["nome_pessoa"]][e["participacao"]] = prob_corrected
            # jump to next inference in the iterator
            continue
        else:
            for info_type_redundant in inf_dict.keys():

                if info_type_redundant in sinonimos_ner:
                    info_type = sinonimos_ner[info_type_redundant]
                else:
                    info_type = info_type_redundant

                # print('\t', info_type_redundant, '->', info_type, file=sys.stderr)

                if not info_type in context:
                    context[info_type] = {}

                all_values = context[info_type]

                if type(inf_dict[info_type_redundant]) == list:
                    infos_iter = inf_dict[info_type_redundant]
                    # print('list', file=sys.stderr)
                elif type(inf_dict[info_type_redundant]) == str:
                    infos_iter = [[inf_dict[info_type_redundant], None]]
                    # print('str', file=sys.stderr)
                elif (
                    type(inf_dict[info_type_redundant]) == float
                    or type(inf_dict[info_type_redundant]) == int
                ):
                    infos_iter = [[float(inf_dict[info_type_redundant]), None]]
                else:
                    # print('Wrong type', file=sys.stderr)
                    break

                for e in infos_iter:
                    if len(e) == 2:
                        a, b = e
                    else:
                        raise Exception(f"Expected list of 2 elements, got {e}")
                    if b is None and type(a) == float:
                        context[info_type] = a
                    else:
                        if type(b) == str:
                            local_value = b
                            prob = a
                        elif type(b) == int or type(b) == float or b == None:
                            local_value = a
                            prob = b
                        else:
                            # print('Wrong types', file=sys.stderr)
                            # print(infos_iter, file=sys.stderr)
                            quit(1)

                        if local_value in all_values:
                            all_values[local_value]["prob"].append(prob)
                            all_values[local_value]["place"] = place
                        else:
                            all_values[local_value] = {"prob": [prob], "place": place}
                        # print('\tUpdated:', local_value, all_values[local_value], file=sys.stderr)

    """print('After first parsing', file=sys.stderr)
    for key, val in context.items():
        print(key, val, file=sys.stderr)"""

    # calcular média de probabilidades
    for info_type, vals_dict in context.items():
        if type(vals_dict) == dict:
            for val in list(vals_dict.keys()):
                val_stats = vals_dict[val]
                probs = val_stats["prob"]
                val_stats["prob"] = sum(
                    [p if p is not None else 0.0 for p in probs]
                ) / len(probs)

    sorted_context = {}
    for info_type in list(context.keys()):
        info_vals = context[info_type]
        if type(info_vals) == dict:
            vals_to_sort = [
                ((d["prob"] + d["place"]) / 2, v) for v, d in info_vals.items()
            ]
            vals_to_sort.sort(reverse=True)
            values_list = [{"value": v, "points": p} for p, v in vals_to_sort]
            sorted_context[info_type] = values_list
        elif type(info_vals) == float or type(info_vals) == int:
            sorted_context[info_type] = info_vals

    """print('After second parsing', file=sys.stderr)
    for key, val in sorted_context.items():
        print(key, val, file=sys.stderr)"""

    sorted_context["envolvimentos"] = envolvimentos
    not_null_keys = [
        "descricao_breve",
        "outras_observacoes",
        "classificacoes_provaveis",
        "classificacao_decisiva",
    ]
    for c in not_null_keys:
        if not c in sorted_context:
            sorted_context[c] = []

    if "nome_do_solicitante" in sorted_context and "pessoa" in sorted_context:
        if (
            len(sorted_context["nome_do_solicitante"]) == 0
            and len(sorted_context["pessoa"]) > 0
        ):
            sorted_context["nome_do_solicitante"] = sorted_context["pessoa"]

    print(
        json.dumps(sorted_context["envolvimentos"], indent=4, ensure_ascii=False),
        file=sys.stderr,
    )

    return sorted_context


def test_summarize_context():
    test_contexts = [
        {
            "rua_ou_logradouro": [["BR", 61.669], ["BR-262", 59.975]],
            "bairro": [["Campos Altos", 89.55]],
            "municipio": [],
            "ponto_de_referencia": [],
            "nome_do_solicitante": [],
            "pessoa": [
                ["Renan", 76.481],
                ["moço", 72.695],
                ["ele", 63.438],
                ["pessoa", 56.435],
            ],
            "numero": [],
            "complemento": [],
        },
        {
            "descricao_breve": "Homem caiu, bateu a cabeça e está tonto/sangrando levemente; pedido de ajuda próximo a BR-262, em Campos Altos, loja H, com orientação para não mexer na cabeça.",
            "outras_observacoes": "Solicitante é Renan. A vítima está consciente, porém zonza e com sangramento no couro cabeludo leve; orientado a não mexer na cabeça e a permanecer imóvel. Local próximo a posto de gasolina na BR-262, Campos Altos, loja H.",
        },
        {
            "classificacoes_provaveis": [
                [0.44863071549401345, "Remoção de Cadáver Vítima de Queda"],
                [0.43430546807430925, "Acidente Vascular Cerebral"],
                [0.43366474101641284, "Acidente de Trânsito Com Vítima"],
                [
                    0.4301109160559612,
                    "Perícia Em Local de Acidente de Trânsito Com Veículo Oficial (vítima Fatal / Parcial)",
                ],
                [
                    0.4176479628844673,
                    "Perícia Em Local de Acidente de Trânsito (vítima Fatal)",
                ],
                [
                    0.40638954306272784,
                    "Remoção de Cadáver Por Sufocamento Ou Esganadura",
                ],
                [0.40330015818137604, "Acidente Aéreo Com Vítimas"],
                [0.4019057813110896, "Pessoa Arrastada Por Enxurrada"],
                [0.3984365116511897, "Acidente de Trânsito Sem Vítima"],
                [0.3974944295798613, "Morte Acidental"],
                [0.3928738541336143, "Acidente Diversos - Trauma"],
                [0.3915539868244673, "Traumatismo Crânio Encefálico"],
                [
                    0.39099917440521154,
                    "Perícia Em Local de Acidente de Trânsito (vítima Parcial)",
                ],
                [0.3883393900294329, "Violação de Sepultura"],
                [0.37587204980210104, "Suspeita de Traumatismo Cranioencefálico (tce)"],
                [0.3745479154070105, "Acidente de Trânsito Com Resultado Morte"],
                [
                    0.372332956980126,
                    "Morte Acidental Provocada Por Eletroplessão (choque Elétrico)",
                ],
                [0.37231310458995487, "Remoção de Cadáver Por Acidente de Trânsito"],
                [0.3715562998443906, "Acidente Aquático Com Vítimas"],
                [
                    0.3658789483921673,
                    "Perícia Em Local de Acidente de Trânsito Com Veículo Oficial (dano)",
                ],
            ]
        },
        {
            "rua_ou_logradouro": [["BR", 61.669], ["BR-262", 59.975]],
            "bairro": [["Campos Altos", 89.55]],
            "municipio": [],
            "ponto_de_referencia": [],
            "nome_do_solicitante": [],
            "pessoa": [["Renan", 75.992], ["moço", 72.695], ["ele", 58.477]],
            "numero": [],
            "complemento": [],
        },
        {"classificacao_decisiva": "Acidente Vascular Cerebral"},
        {
            "rua_ou_logradouro": [["BR", 61.669], ["BR-262", 59.975]],
            "bairro": [["Campos Altos", 89.55]],
            "municipio": [],
            "ponto_de_referencia": [],
            "nome_do_solicitante": [],
            "pessoa": [["Renan", 75.738], ["moço", 72.695], ["ele", 57.901]],
            "numero": [],
            "complemento": [],
        },
        {
            "descricao_breve": "Acidente com queda de homem que bateu cabeça, aparenta tontura; ferimento leve no couro cabeludo; necessidade de evitar movimentar a vítima e aguardar atendimento próximo a BR-262, Campos Altos.",
            "outras_observacoes": "Vítima consciente porém zonza; não permitir que a vítima se mova ou levante; há possibilidade de sangramento leve no cabelo; telefoneador é Renan; localização aproximada: loja H, próximo ao posto na BR-262, Campos Altos; instruções repassadas para manter a calma e aguardar socorro.",
        },
        {
            "classificacoes_provaveis": [
                [0.7075986815159205, "Acidente de Trânsito Com Vítima"],
                [0.6682093432265557, "Acidente de Trânsito Sem Vítima"],
                [0.6542924359594209, "Acidente Diversos - Trauma"],
                [
                    0.6448315098013803,
                    "Perícia Em Local de Acidente de Trânsito Com Veículo Oficial (vítima Fatal / Parcial)",
                ],
                [0.6251717407435455, "Morte Acidental"],
                [0.6120263205829581, "Acidente de Trânsito Com Resultado Morte"],
                [
                    0.6022553224859984,
                    "Perícia Em Local de Acidente de Trânsito (vítima Fatal)",
                ],
                [0.5938291798287006, "Acidente Envolvendo Veículo Oficial"],
                [0.5879967165224353, "Remoção de Cadáver Por Acidente de Trânsito"],
                [
                    0.578637637352244,
                    "Perícia Em Local de Acidente de Trânsito (vítima Parcial)",
                ],
                [0.5784611757047466, "Remoção de Cadáver Vítima de Queda"],
                [0.5755960365913303, "Outro Tipo de Acidente Com Veículo"],
                [0.5749924257474825, "Acidente Aéreo Com Vítimas"],
                [0.5721230428748865, "Acidente Aquático Com Vítimas"],
                [0.5625236749570914, "Suspeita de Traumatismo Cranioencefálico (tce)"],
                [0.5582468293812926, "Acidentes Naturais"],
                [
                    0.555471802606746,
                    "Perícia Em Local de Acidente de Trânsito Com Veículo Oficial (dano)",
                ],
                [0.5497600494229526, "Acidente Vascular Cerebral"],
                [0.5450870336640701, "Traumatismo Crânio Encefálico"],
                [0.5409774355377672, "Suspeita de Acidente Vascular Encefálico (ave)"],
            ]
        },
        {"classificacao_decisiva": "Acidente Diversos - Trauma"},
    ]
    context_types = [
        "ner",
        "descricao_e_observacao",
        "lista_de_naturezas",
        "ner",
        "natureza_decisiva",
        "ner",
        "descricao_e_observacao",
        "lista_de_naturezas",
        "natureza_decisiva",
    ]

    inferences = [
        {"tipo_de_inferencia": t, "resultado": r}
        for t, r in zip(context_types, test_contexts)
    ]

    c = summarize_context(inferences)
    print(json.dumps(c, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_summarize_context()
