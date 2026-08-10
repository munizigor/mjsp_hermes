from os import path
import sys
import json
import gc
import numpy as np
from torch import cuda
import polars as pl
import matplotlib.pyplot as plt
from scipy.spatial import distance

from tqdm import tqdm
from not_used.main_interpreter import ollama_interpret_call_transcript, stop_ollama
from ner.gline_interpreter import extract_labels, load_gliner_cuda
from ner.plotting import plot_metainfo
from ner.string_encoding import SentenceEncodingCache
from ner.util import multi_preds_to_list

def run_ner(interpretations_json_path, n_to_test_default, 
            fake_calls_json_dataset):
    interpretations = [
        {'input': fake_calls_json_dataset[index],
         'entities': {
             'gliner': None, 
             'gaia': None,
         },
        'response_meta': {
            'gliner': None, 
            'gaia': None},
        }
        for index in range(min(len(fake_calls_json_dataset), n_to_test_default))
    ]

    try:
        stop_ollama()
    except Exception as e:
        pass

    gliner_model = load_gliner_cuda()
    n_to_test = n_to_test_default
    for i in tqdm(range(n_to_test)):
        if i >= len(fake_calls_json_dataset):
            break
        input_data = interpretations[i]['input']
        #print(f"Interpreting call {i + 1} of {n_to_test_default}")
        transcript = input_data['roteiro']
        result, response_meta = extract_labels(transcript, gliner_model)
        #print('Call metadata:')
        #print(json.dumps(input_data['Emergencia'], indent=4, ensure_ascii=False))
        #print(json.dumps(input_data['Perfil do Solicitante'], indent=4, ensure_ascii=False))
        #print(f"Result for call {i + 1}: {result}")
        print(f"Response metadata for call {i + 1}: {response_meta}")
        print('---')
        # Store the interpretation result along with the input and response metadata
        interpretations[i]['entities']['gliner'] = result
        interpretations[i]['response_meta']['gliner'] = response_meta

    json.dump(interpretations, open(interpretations_json_path, 'w'), indent=4, ensure_ascii=False)

    # Clear memory
    del gliner_model
    cuda.empty_cache()
    gc.collect()
    
    n_to_test = n_to_test_default
    for i in tqdm(range(n_to_test)):
        if i >= len(fake_calls_json_dataset):
            break
        input_data = interpretations[i]['input']
        print(f"Interpreting call {i + 1} of {n_to_test_default}")
        transcript = input_data['roteiro']
        result, response_meta = ollama_interpret_call_transcript(transcript)
        print('Call metadata:')
        print(json.dumps(input_data['Emergencia'], indent=4, ensure_ascii=False))
        print(json.dumps(input_data['Perfil do Solicitante'], indent=4, ensure_ascii=False))
        print(f"Result for call {i + 1}: {result}")
        print(f"Response metadata for call {i + 1}: {response_meta}")
        print('---')
        # Store the interpretation result along with the input and response metadata
        interpretations[i]['entities']['gaia'] = result
        interpretations[i]['response_meta']['gaia'] = response_meta

        json.dump(interpretations, open(interpretations_json_path, 'w'), indent=4, ensure_ascii=False)
    stop_ollama()
    # Clear memory
    cuda.empty_cache()
    gc.collect()

    return interpretations

def create_truth_df(fake_calls_dataset, to_load = 100):
    lines = []
    index = 0

    for fake_call in fake_calls_dataset:
        profile = fake_call['Perfil do Solicitante']
        addr = fake_call['Emergencia']['Endereco']
        #natureza = fake_call['input']['Emergencia']['Natureza']
        new_line = {
            'id': index,
            'nome_do_solicitante': profile['Nome Solicitante'],
            'telefone_ou_celular': profile['Numero'],
            'ponto_de_referencia': addr['ref_name']
        }

        for addr_part in ['rua', 'numero', 'bairro', 'cidade']:
            if addr_part in addr['desc_tipo']:
                new_line[addr_part] = addr[addr_part]
            else:
                new_line[addr_part] = ""
        lines.append(new_line)
            
        index += 1
        if to_load == index:
            break
    
    truth_df = pl.DataFrame(lines)
    #replace "" (empty string) with None
    for col in truth_df.columns:
        truth_df = truth_df.with_columns(pl.col(pl.String).replace("", None))
    return truth_df


def create_pred_df(interpretations):
    vals_to_ignore = ['None', 'none', 'N/A', 'n/a', 'null', 'NULL', '']
    vals_to_ignore += ['Não especificado', 'não especificado', 'não informado', 'não informado']
    vals_to_ignore += ['Não informado', 'não informado', 'não informado']
    vals_to_ignore += ['cidade', 'Cidade', 'Bairro', 'bairro', 'Apartamento', 'apartamento', 
                       'Rua', 'rua', 'Número', 'numero', 'número', 'rapaz', 'banhista', 'Hesitante',
                       'homem', 'mulher', 'senhor', 'senhora', 'pessoa', 'Pessoa',
                       'solicitante', 'Solicitante', 'galho', 'árvore', 'piscina', 'entrada', 'pátio',
                       'floresta', 'chão', 'estrada', 'prédio']
    vals_to_ignore = set(vals_to_ignore)
    index = 0

    key_translate = {'cidade': 'municipio', 'numero': 'numero_na_rua', 'rua': 'rua_ou_logradouro'}
    df_lines = {'gliner': [], 'gaia': []}

    for fake_call in interpretations:
        preds = fake_call['entities']

        for model_name in ['gliner', 'gaia']:
            model_preds = preds[model_name]
            new_line = {
                'id': index
            }
            if 'nome_do_solicitante' in model_preds:
                new_line['nome_do_solicitante'] = multi_preds_to_list(model_preds['nome_do_solicitante'])
            if 'ponto_de_referencia' in model_preds:
                new_line['ponto_de_referencia'] = multi_preds_to_list(model_preds['ponto_de_referencia'])
            elif 'ponto_referencia' in model_preds:
                new_line['ponto_de_referencia'] = multi_preds_to_list(model_preds['ponto_referencia'])

            for addr_part in ['rua', 'numero', 'bairro', 'cidade']:
                addr_part_trans = key_translate[addr_part] if addr_part in key_translate else addr_part
                if addr_part_trans in model_preds:
                    new_line[addr_part] = multi_preds_to_list(model_preds[addr_part_trans])

            for key in new_line.keys():
                if key != 'id':
                    if isinstance(new_line[key], int) or isinstance(new_line[key], float):
                        new_line[key] = str(new_line[key])
                    if len(new_line[key]) == 0:
                        new_line[key] = None
                    if new_line[key] != None:
                        new_line[key] = [v for v in new_line[key] if not v in vals_to_ignore]

            df_lines[model_name].append(new_line)
        index += 1
    
    dfs = {
        model_name: pl.DataFrame(lines)
        for model_name, lines in df_lines.items()
    }

    return dfs

def create_metainfo_df(interpretations):
    lines = []
    index = 0

    for fake_call in interpretations:
        response_meta = fake_call['response_meta']
        gaia_line = {'id': index}
        if response_meta['gaia'] != None:
            for k, v in response_meta['gaia'].items():
                if k == 'model':
                    gaia_line['version'] = v.split(':')[-1]
                    gaia_line[k] = v.split(':')[0]
                else:
                    gaia_line[k] = v

        gliner_line = {'id': index}
        if response_meta['gliner'] != None:
            for k, v in response_meta['gliner'].items():
                if k == 'request_duration':
                    k = 'result_time'
                gliner_line[k] = v
        
        lines += [gaia_line, gliner_line]
        index += 1
    
    df = pl.DataFrame(lines)
    return df

if __name__ == "__main__":
    print(sys.argv)
    if 'kernel' in str(sys.argv) or len(sys.argv) == 1:
        fake_calls_json_dataset_path = '../fake-calls/generated/cnmoro-gemma3-gaia-ptbr-4b_q8_0/chamadas_roteirizadas_final.json'
        interpretations_json_path = 'results/gline_gaia_3/550_test.json'
        n_to_test_default = 100
    else:
        fake_calls_json_dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'chamadas_roteirizadas_final.json'
        interpretations_json_path = sys.argv[2] if len(sys.argv) > 2 else 'test.json'
        n_to_test_default = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    print(n_to_test_default)
    fake_calls_json_dataset = json.load(open(fake_calls_json_dataset_path, 'r'))

    if not path.exists(interpretations_json_path):
        interpretations = run_ner(interpretations_json_path, n_to_test_default, 
            fake_calls_json_dataset)
    else:
        interpretations = json.load(open(interpretations_json_path, 'r'))
    
    truth_df = create_truth_df(fake_calls_json_dataset, to_load=n_to_test_default)
    metainfo_df = create_metainfo_df(interpretations)
    plot_metainfo(metainfo_df, interpretations_json_path.replace('.json', ''))
    # Save the metainfo dataframe
    metainfo_df.write_csv(interpretations_json_path.replace('.json', '.metainfo.csv'))

    pred_dfs = create_pred_df(interpretations)
    gliner_df = pred_dfs['gliner']
    gaia_df = pred_dfs['gaia']

    # Save the truth and predictions dataframes
    truth_df.write_parquet(interpretations_json_path.replace('.json', '.truth.parquet'))
    gliner_df.write_parquet(interpretations_json_path.replace('.json', '.gliner.parquet'))
    gaia_df.write_parquet(interpretations_json_path.replace('.json', '.gaia.parquet'))

    print("Truth DataFrame:")
    print(truth_df)
    print("Gliner DataFrame:")
    print(gliner_df)
    print("Gaia DataFrame:")
    print(gaia_df)

    pred_keys = ['rua', 'numero', 'bairro', 'cidade', 'nome_do_solicitante', 'ponto_de_referencia']

    encoder_cache = SentenceEncodingCache(
        model_name='PORTULAN/serafim-100m-portuguese-pt-sentence-encoder'
    )

    comparisons = []

    sim_threshold = 0.75

    for info_col in tqdm(pred_keys):
        #get indexes where truth_df[info_col] is not None
        truth_filtered = truth_df.filter(pl.col(info_col).is_not_null())
        not_none_ids = truth_filtered.get_column('id').to_list()
        gliner_df_filtered = gliner_df.filter(pl.col('id').is_in(not_none_ids))
        gaia_df_filtered = gaia_df.filter(pl.col('id').is_in(not_none_ids))

        truth_vals = truth_filtered.get_column(info_col).to_list()
        gliner_vals = gliner_df_filtered.get_column(info_col).to_list()
        gaia_vals = gaia_df_filtered.get_column(info_col).to_list()

        for i in range(len(truth_vals)):
            if gliner_vals[i] == None or gliner_vals[i] != gliner_vals[i]:
                gliner_vals[i] = []
            if gaia_vals[i] == None or gaia_vals[i] != gaia_vals[i]:
                gaia_vals[i] = []

        values_for_encoding = set()
        for val in truth_vals + gliner_vals + gaia_vals:
            if isinstance(val, list):
                values_for_encoding.update(val)
            elif val is not None:
                values_for_encoding.add(val)
        
        values_for_encoding = list(values_for_encoding)
        encoder_cache.pre_calc(values_for_encoding)
            
        semantic_sims = []
        for i in range(len(truth_vals)):
            truth_val = truth_vals[i]
            gliner_val = gliner_vals[i]
            gaia_val = gaia_vals[i]

            truth_float = encoder_cache.get_encoding(truth_val)
            gliner_float = [encoder_cache.get_encoding(v) for v in gliner_val]
            gaia_float = [encoder_cache.get_encoding(v) for v in gaia_val]

            if len(gliner_float) == 0:
                gliner_sim = None
            else:
                gliner_sims = encoder_cache.model.similarity(
                    [truth_float], gliner_float
                ).tolist()[0]
                gliner_sim = max(gliner_sims)
            if len(gaia_float) == 0:
                gaia_sim = None
            else:
                gaia_sims = encoder_cache.model.similarity(
                    [truth_float], gaia_float
                ).tolist()[0]
                gaia_sim = max(gaia_sims)
            
            gliner_result = 'FP'
            gaia_result = 'FP'
            if gliner_sim == None:
                gliner_result = 'FN'
            elif gliner_sim >= sim_threshold:
                gliner_result = 'TP'
            
            if gaia_sim == None:
                gaia_result = 'FN'
            elif gaia_sim >= sim_threshold:
                gaia_result = 'TP'
            
            id_field = not_none_ids[i]
            semantic_sims.append({
                'id': id_field,
                'truth': truth_val,
                'gliner': gliner_val,
                'gaia': gaia_val,
                'gliner_sim': gliner_sim,
                'gaia_sim': gaia_sim,
                'gliner_result': gliner_result,
                'gaia_result': gaia_result 
            })
        
        no_truth = truth_df.filter(pl.col(info_col).is_null())
        no_truth_ids = no_truth.get_column('id').to_list()
        gliner_df_filtered = gliner_df.filter(pl.col('id').is_in(no_truth_ids))
        gaia_df_filtered = gaia_df.filter(pl.col('id').is_in(no_truth_ids))
        
        gliner_vals = gliner_df_filtered.get_column(info_col).to_list()
        gaia_vals = gaia_df_filtered.get_column(info_col).to_list()

        for i in range(len(gliner_vals)):
            if gliner_vals[i] == None or gliner_vals[i] != gliner_vals[i]:
                gliner_vals[i] = []
        for i in range(len(gaia_vals)):
            if gaia_vals[i] == None or gaia_vals[i] != gaia_vals[i]:
                gaia_vals[i] = []

        values_for_encoding = set()
        for val in gliner_vals + gaia_vals:
            if isinstance(val, list):
                values_for_encoding.update(val)
            elif val is not None:
                values_for_encoding.add(val)
        
        values_for_encoding = list(values_for_encoding)
        encoder_cache.pre_calc(values_for_encoding)
        
        for i in range(len(gliner_vals)):
            gliner_val = gliner_vals[i]
            gaia_val = gaia_vals[i]
            
            gliner_result = 'TN' if len(gliner_vals[i]) == 0 else 'FP'
            gaia_result = 'TN' if len(gaia_vals[i]) == 0 else 'FP'
            
            id_field = no_truth_ids[i]
            semantic_sims.append({
                'id': id_field,
                'truth': None,
                'gliner': gliner_val,
                'gaia': gaia_val,
                'gliner_sim': None,
                'gaia_sim': None,
                'gliner_result': gliner_result,
                'gaia_result': gaia_result
            })

        semantic_sims_df = pl.DataFrame(semantic_sims)

        semantic_sims_df.write_parquet(interpretations_json_path.replace(
            '.json', f'.{info_col}_semantic_sims.parquet'))
        comparisons.append([info_col, semantic_sims_df])
    cidades = [df for n, df in comparisons if n == 'cidade'][0]
    bairros = [df for n, df in comparisons if n == 'bairro'][0]
    stat_lines = []
    for info_col, semantic_sims_df in comparisons:
        gliner_sims = semantic_sims_df.get_column('gliner_sim').to_list()
        gaia_sims = semantic_sims_df.get_column('gaia_sim').to_list()
        gliner_sims = np.array([s for s in gliner_sims if s is not None and s == s])
        gaia_sims = np.array([s for s in gaia_sims if s is not None and s == s])

        gliner_mean = gliner_sims.mean()
        gaia_mean = gaia_sims.mean()
        gliner_std = gliner_sims.std()
        gaia_std = gaia_sims.std()

        #get TP, TN, FP, FN counts for gaia
        gaia_results = semantic_sims_df.get_column('gaia_result').to_list()
        gaia_tp = gaia_results.count('TP')
        gaia_tn = gaia_results.count('TN')
        gaia_fp = gaia_results.count('FP')
        gaia_fn = gaia_results.count('FN')

        #get TP, TN, FP, FN counts for gliner
        gliner_results = semantic_sims_df.get_column('gliner_result').to_list()
        gliner_tp = gliner_results.count('TP')
        gliner_tn = gliner_results.count('TN')
        gliner_fp = gliner_results.count('FP')
        gliner_fn = gliner_results.count('FN')

        #Calc precision, recall, f1 for gaia
        gaia_precision = gaia_tp / (gaia_tp + gaia_fp) if (gaia_tp + gaia_fp) > 0 else 0
        gaia_recall = gaia_tp / (gaia_tp + gaia_fn) if (gaia_tp + gaia_fn) > 0 else 0
        gaia_f1 = (2 * gaia_precision * gaia_recall) / (gaia_precision + gaia_recall) if (gaia_precision + gaia_recall) > 0 else 0

        #Calc precision, recall, f1 for gliner
        gliner_precision = gliner_tp / (gliner_tp + gliner_fp) if (gliner_tp + gliner_fp) > 0 else 0
        gliner_recall = gliner_tp / (gliner_tp + gliner_fn) if (gliner_tp + gliner_fn) > 0 else 0
        gliner_f1 = (2 * gliner_precision * gliner_recall) / (gliner_precision + gliner_recall) if (gliner_precision + gliner_recall) > 0 else 0

        stat_lines.append({
            'info_col': info_col,
            'gliner_mean': round(gliner_mean, 3),
            'gaia_mean': round(gaia_mean, 3),
            'gliner_std': round(gliner_std, 3),
            'gaia_std': round(gaia_std, 3),
            'gliner_precision': round(gliner_precision, 3),
            'gaia_precision': round(gaia_precision, 3),
            'gliner_recall': round(gliner_recall, 3),
            'gaia_recall': round(gaia_recall, 3),
            'gliner_f1': round(gliner_f1, 3),
            'gaia_f1': round(gaia_f1, 3)
        })
    
    stat_df = pl.DataFrame(stat_lines)
    stat_df.write_excel(interpretations_json_path.replace('.json', '.semantic_sims_stats.xlsx'))

    encoder_cache.close()