import sys
import polars as pl

from data_processing import save_to_parquet

output_dir = sys.argv[1]
#output_dir = '/home/pita/docs/hermes/hermes-agents/results/o1-tts_azure_speech-cl_5.0-17-09-2025_18:17:00/'

env_vals = {rawline.split('=')[0]: rawline.split('=')[1].rstrip('\n')
        for rawline in open('.env', 'r').read().split('\n')  if '=' in rawline}
sqlite_path = env_vals['LOCAL_SQL_DB_PATH']+'/database.sqlite'

res = save_to_parquet(sqlite_path, output_dir, {})
if res != None:
    err, emergencias_df, transcricao_df, audios_df, inferencia_df = res
    if str(emergencias_df.dtypes[-1]) == 'Object':
        emergencias_df = emergencias_df.with_columns(delay_local = emergencias_df['delay_local'].cast(str))
    if str(emergencias_df.dtypes[-2]) == 'Object':
        emergencias_df = emergencias_df.with_columns(dataset_index = emergencias_df['dataset_index'].cast(str))
    
    emergencias_df.write_parquet(output_dir+'/emergencias_df.parquet')