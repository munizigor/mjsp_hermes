import sys
import os
from glob import glob
import json

import pandas as pd
import numpy as np

cost_models_dir = os.path.join(os.path.dirname(__file__), 'costs_models')
train_data_dir = os.path.join(os.path.dirname(__file__), 'train_data')

load_tests_dir = sys.argv[1]
results_paths = glob(f'{load_tests_dir}/*/results.xlsx')

call_data_dfs = []
token_counts_by_infer_dfs = []
verbosity_dfs = []
db_sizes = []

print("Analysis Results found:", len(results_paths))

for p in results_paths:
    test_dir = os.path.dirname(p)
    print("Test directory:", test_dir)
    test_name = os.path.basename(test_dir)

    sqlitedb_path = f'{test_dir}/sqlite.db'
    db_file_size = os.path.getsize(sqlitedb_path)
    db_file_size_mb = db_file_size / 1024 / 1024
    
    df_em = pd.read_excel(p, sheet_name='Emergencias Df')
    df_infer = pd.read_excel(p, sheet_name='Inferencia Df')

    modelos = df_infer['modelo_utilizado'].unique()
    if 'gliner_gliclass' in modelos:
        continue

    emergency_context_length_df = []

    for em_id, em_df in df_infer[df_infer['tipo_de_inferencia'] == 'asr'].groupby('id_emergencia'):
        em_df = em_df.sort_values('horario_contexto')
        n_chars = 0
        context_lens = []
        for _, row in em_df.iterrows():
            n_chars += len(row['resultado'])
            horario_contexto = row['horario_contexto']
            context_lens.append({'id_emergencia': em_id, 'horario_contexto': horario_contexto, 'transcription_len': n_chars})
        emergency_context_length_df.append(pd.DataFrame(context_lens))

    emergency_context_length_df = pd.concat(emergency_context_length_df, ignore_index=True)

    def get_context_length_at_timestamp(em_id, timestamp):
        #print(emergency_context_length_df['id_emergencia'])
        #print(em_id, type(em_id))
        emergency_df = emergency_context_length_df[emergency_context_length_df['id_emergencia'] == em_id]
        #print(emergency_df)
        #print(timestamp, type(timestamp))
        filtered = emergency_df[emergency_df['horario_contexto'] <= timestamp]
        #print(filtered)
        if filtered.empty:
            #print(0)
            return 0
        #print(filtered['transcription_len'].max())
        return filtered['transcription_len'].max()
    
    verbosity_rows = []
    for _, row in df_infer[df_infer['tipo_de_inferencia'] != 'asr'].iterrows():
        em_id = row['id_emergencia']
        timestamp = row['horario_contexto']
        prompt_len = get_context_length_at_timestamp(em_id, timestamp)
        task = row['tipo_de_inferencia']
        tokens_in = row['input_tokens']
        tokens_out = row['output_tokens']
        answer_len = len(row['resultado'])
        
        verbosity_rows.append({'modelo_utilizado': row['modelo_utilizado'], 'tipo_de_inferencia': task, 'id_emergencia': em_id,  
            'horario_contexto': timestamp, 'prompt_len': prompt_len, 'answer_len': answer_len, 
            'tokens_in': tokens_in, 'tokens_out': tokens_out})
    
    verbosity_df = pd.DataFrame(verbosity_rows)
    verbosity_dfs.append(verbosity_df)

    
    token_counts_by_infer = []
    ner_by_model = {}
    for m in modelos:
        for ner_type, ner_df in df_infer[df_infer['modelo_utilizado'] == m].groupby('tipo_de_inferencia'):
            ner_by_model[ner_type] = m
            tokens_in = ner_df['input_tokens'].sum()
            tokens_out = ner_df['output_tokens'].sum()
            row = {'modelo': m, 'ner_type': ner_type, 'tokens_in': tokens_in, 'tokens_out': tokens_out}
            token_counts_by_infer.append(row)
    token_counts_by_infer_df = pd.DataFrame(token_counts_by_infer)
    token_counts_by_infer_df['test_name'] = test_name
    
    #comprimento_total_audios is in seconds
    call_data = df_em[['nome_no_dataset', 'comprimento_total_audios', 'tokens_in', 'tokens_out']]
    for ner_type, model in ner_by_model.items():
        call_data[f'modelo_{ner_type}'] = model
    
    call_data['test_name'] = test_name
    print(call_data.shape)
    call_data_dfs.append(call_data.copy())

    total_test_audio_secs = call_data['comprimento_total_audios'].sum()
    token_counts_by_infer_df['test_audio_secs'] = total_test_audio_secs

    token_counts_by_infer_dfs.append(token_counts_by_infer_df.copy())

    db_sizes.append({'test_name': test_name, 'db_size_mb': db_file_size_mb, 'total_test_audio_secs': total_test_audio_secs})

#1. Find verbosity and token granularity

verbosity_df = pd.concat(verbosity_dfs, ignore_index=True)

#tokens_in cant be null, but tokens_out can
verbosity_df = verbosity_df[verbosity_df['tokens_in'].notna()]
verbosity_df['input_granularity'] =  verbosity_df['prompt_len'] / verbosity_df['tokens_in'] 
verbosity_df['output_granularity'] = verbosity_df['answer_len'] / verbosity_df['tokens_out']
verbosity_df['expansion_ratio'] = verbosity_df['tokens_out'] / verbosity_df['tokens_in']

print(verbosity_df)
print(verbosity_df.describe())
verbosity_df.to_csv(os.path.join(train_data_dir, 'verbosity.csv'), index=False)

model_verbosity_df = []
stats_by_model = {}
for model, model_df in verbosity_df.groupby('modelo_utilizado'):
    granularities = model_df['input_granularity'] + model_df['output_granularity']
    mean_granularity = granularities.mean()

    expansion_ratios = []
    for task, task_df in model_df.groupby('tipo_de_inferencia'):
        mean_exp_ratio = task_df['expansion_ratio'].mean()
        expansion_ratios.append(mean_exp_ratio)
    mean_expansion_ratio = np.mean(expansion_ratios)

    row = {'modelo_utilizado': model, 'mean_granularity': mean_granularity, 
        'mean_expansion_ratio': mean_expansion_ratio}
    stats_by_model[model] = {'mean_granularity': mean_granularity,
        'mean_expansion_ratio': mean_expansion_ratio}
    model_verbosity_df.append(row)

model_verbosity_df = pd.DataFrame(model_verbosity_df)
print(model_verbosity_df)
print(model_verbosity_df.describe())
model_verbosity_df.to_csv(os.path.join(train_data_dir, 'model_verbosity.csv'), index=False)

median_token_granularity = model_verbosity_df['mean_granularity'].median()
print('Median token granularity:', median_token_granularity)
median_expansion_ratio = model_verbosity_df['mean_expansion_ratio'].median()
print('Median expansion ratio:', median_expansion_ratio)

#2. Token count normalization
#Input tokens must be normalized relative to the median_token_granularity
#Output tokens must be normalized both by median_token_granularity and median_expansion_ratio
#High token counts of very granular or very verbose models will be deflated, while low token counts of models with low verbosity will be inflated

granularity_map = {k: v['mean_granularity'] for k, v in stats_by_model.items()}
expansion_map = {k: v['mean_expansion_ratio'] for k, v in stats_by_model.items()}

token_counts_by_infer_df = pd.concat(token_counts_by_infer_dfs, ignore_index=True)

model_granularity = token_counts_by_infer_df['modelo'].map(granularity_map)
model_expansion = token_counts_by_infer_df['modelo'].map(expansion_map)

# Scale input by granularity (deflates models that output too many tokens per character)
token_counts_by_infer_df['tokens_in_norm'] = token_counts_by_infer_df['tokens_in'] * (model_granularity / median_token_granularity)

# Scale output by granularity, then inversely by expansion ratio (deflates verbose models)
token_counts_by_infer_df['tokens_out_norm'] = token_counts_by_infer_df['tokens_out'] * (model_granularity / median_token_granularity) * (median_expansion_ratio / model_expansion)

token_counts_by_infer_df['avg_input_tokens_per_sec'] = token_counts_by_infer_df['tokens_in'] / token_counts_by_infer_df['test_audio_secs']
token_counts_by_infer_df['avg_input_tokens_norm_per_sec'] = token_counts_by_infer_df['tokens_in_norm'] / token_counts_by_infer_df['test_audio_secs']
token_counts_by_infer_df['avg_output_tokens_per_sec'] = token_counts_by_infer_df['tokens_out'] / token_counts_by_infer_df['test_audio_secs']
token_counts_by_infer_df['avg_output_tokens_norm_per_sec'] = token_counts_by_infer_df['tokens_out_norm'] / token_counts_by_infer_df['test_audio_secs']

print(token_counts_by_infer_df.describe())
token_counts_by_infer_df.to_csv(os.path.join(train_data_dir, 'token_counts_by_infer.csv'), index=False)

call_data_df = pd.concat(call_data_dfs, ignore_index=True)

model_cols = [c for c in call_data_df.columns if c.startswith('modelo_')]
    
if model_cols:
    # Average the metrics if multiple models were used for different NER tasks during the call
    call_granularities = pd.concat([call_data_df[col].map(granularity_map) for col in model_cols], axis=1).mean(axis=1)
    call_expansions = pd.concat([call_data_df[col].map(expansion_map) for col in model_cols], axis=1).mean(axis=1)
    
    call_data_df['tokens_in_norm'] = call_data_df['tokens_in'] * (call_granularities / median_token_granularity)
    call_data_df['tokens_out_norm'] = call_data_df['tokens_out'] * (call_granularities / median_token_granularity) * (median_expansion_ratio / call_expansions)

print(call_data_df)
print(call_data_df.describe())

call_data_df['in_tokens_per_sec'] = call_data_df['tokens_in'] / call_data_df['comprimento_total_audios']
call_data_df['in_tokens_norm_per_sec'] = call_data_df['tokens_in_norm'] / call_data_df['comprimento_total_audios']
call_data_df['out_tokens_per_sec'] = call_data_df['tokens_out'] / call_data_df['comprimento_total_audios']
call_data_df['out_tokens_norm_per_sec'] = call_data_df['tokens_out_norm'] / call_data_df['comprimento_total_audios']

print(call_data_df.describe())

call_data_df.to_csv(os.path.join(train_data_dir, 'call_tokens.csv'), index=False)

for col in ['in_tokens_per_sec', 'in_tokens_norm_per_sec', 'out_tokens_per_sec', 'out_tokens_norm_per_sec']:
    print(col)
    print(call_data_df[col].mean(), call_data_df[col].std())

#3. MB of DB per hour of audio
db_sizes_df = pd.DataFrame(db_sizes)

db_sizes_df['test_audio_hours'] = db_sizes_df['total_test_audio_secs'] / 60 / 60

db_sizes_df['db_size_mb_per_hour'] = db_sizes_df['db_size_mb'] / db_sizes_df['test_audio_hours']

print(db_sizes_df)
print(db_sizes_df.describe())

db_sizes_df.to_csv(os.path.join(train_data_dir, 'db_sizes.csv'), index=False)

model_simple = {
    "in_tokens_per_hour": call_data_df['in_tokens_norm_per_sec'].mean() * 60 * 60,
    "in_tokens_per_hour_std": call_data_df['in_tokens_norm_per_sec'].std() * 60 * 60,
    "out_tokens_per_hour": call_data_df['out_tokens_norm_per_sec'].mean() * 60 * 60,
    "out_tokens_per_hour_std": call_data_df['out_tokens_norm_per_sec'].std() * 60 * 60,
    "db_size_mb_per_hour": db_sizes_df['db_size_mb_per_hour'].mean(),
    "db_size_mb_per_hour_std": db_sizes_df['db_size_mb_per_hour'].std(),
}

with open(os.path.join(cost_models_dir, 'model_simple.json'), 'w') as f:
    json.dump(model_simple, f, indent=4, ensure_ascii=False)