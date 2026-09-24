import sys
import os
from glob import glob
import json
import time

import requests
import unicodedata

import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
train_data_dir = os.path.join(os.path.dirname(__file__), 'train_data')
dados_coletados = base_dir, "dados_coletados"
cenarios_excel_path = os.path.join(dados_coletados, 'cenarios.xlsx')

def normalizar_nome(nome):
    """Remove acentos e padroniza para minúsculas para facilitar a busca do ID."""
    nome = unicodedata.normalize('NFKD', str(nome)).encode('ASCII', 'ignore').decode('utf-8')
    return nome.lower().strip()

class ExtratorIBGE:
    def __init__(self):
        self.mapa_estados = {}
        self.mapa_municipios = {}
        self.var_id = "2"
        self.class_id = "2"
        self._carregar_localidades()
        self._carregar_metadados_tabela()

    def _carregar_metadados_tabela(self):
        url_meta = "https://servicodados.ibge.gov.br/api/v3/agregados/9923/metadados"
        try:
            meta = requests.get(url_meta, timeout=12).json()
            self.var_id = meta['variaveis'][0]['id']
            for c in meta.get('classificacoes', []):
                if 'domic' in c['nome'].lower():
                    self.class_id = c['id']
                    break
        except Exception as e:
            print(f"Aviso: Erro metadados: {e}")

    def _carregar_localidades(self, tentativas=5):
        import time
        # Mapeia Estados (N3)
        est_url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
        for t in range(tentativas):
            try:
                resp = requests.get(est_url, timeout=15)
                # Só tenta converter se o status for 200 (Sucesso) e o texto não estiver vazio
                if resp.status_code == 200 and resp.text.strip():
                    for uf in resp.json():
                        self.mapa_estados[normalizar_nome(uf['nome'])] = uf['id']
                        self.mapa_estados[normalizar_nome(uf['sigla'])] = uf['id']
                    break
                else:
                    print(f"Aviso: IBGE retornou erro {resp.status_code} na malha de Estados. Tentando novamente...")
                    time.sleep(2)
            except Exception as e:
                print(f"Erro ao buscar estados (tentativa {t+1}): {e}")
                time.sleep(2)

        # Mapeia Municípios (N6)
        mun_url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        for t in range(tentativas):
            try:
                resp = requests.get(mun_url, timeout=15)
                if resp.status_code == 200 and resp.text.strip():
                    for mun in resp.json():
                        self.mapa_municipios[normalizar_nome(mun['nome'])] = mun['id']
                    break
                else:
                    print(f"Aviso: IBGE retornou erro {resp.status_code} na malha de Municípios. Tentando novamente...")
                    time.sleep(2)
            except Exception as e:
                print(f"Erro ao buscar municípios (tentativa {t+1}): {e}")
                time.sleep(2)

    def obter_estimativa_populacao_atual(self):
        """Busca a projeção/estimativa populacional do Brasil em tempo real."""
        url = "https://servicodados.ibge.gov.br/api/v1/projecoes/populacao"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()['projecao']['populacao']
        except Exception as e:
            print(f"Aviso: Erro ao buscar relógio de projeção: {e}")
        return 212583720 # Fallback baseado na última Estimativa Oficial IBGE (Jul/2024)

    def obter_taxa_urbanizacao(self, nome_territorio, tentativas=2):
        nome_norm = normalizar_nome(nome_territorio)
        
        # Agora o Extrator reconhece o "Brasil" diretamente na malha oficial N1
        if nome_norm == 'brasil':
            nivel, codigo = "N1", "1"
        elif nome_norm in self.mapa_estados:
            nivel, codigo = "N3", self.mapa_estados[nome_norm]
        elif nome_norm in self.mapa_municipios:
            nivel, codigo = "N6", self.mapa_municipios[nome_norm]
        else:
            return None

        url = f"https://servicodados.ibge.gov.br/api/v3/agregados/9923/periodos/2022/variaveis/{self.var_id}?localidades={nivel}[{codigo}]&classificacao={self.class_id}[all]"
        
        for tentativa in range(tentativas):
            try:
                import time
                time.sleep(0.3)
                resp = requests.get(url, timeout=12)
                
                if resp.status_code != 200 or not resp.json():
                    continue
                    
                resultados = resp.json()[0]['resultados']
                pop_urbana = 0
                pop_rural = 0
                
                for res in resultados:
                    valor_str = res['series'][0]['serie']['2022']
                    if not valor_str.isdigit(): 
                        continue
                    valor = int(valor_str)
                    
                    for classif in res.get('classificacoes', []):
                        categoria = list(classif['categoria'].values())[0].lower()
                        if 'urbana' in categoria:
                            pop_urbana += valor
                        elif 'rural' in categoria:
                            pop_rural += valor

                pop_total = pop_urbana + pop_rural
                taxa = (pop_urbana / pop_total) if pop_total > 0 else 0
                
                return {
                    'territorio': nome_territorio,
                    'pop_total': pop_total,
                    'pop_urbana': pop_urbana,
                    'taxa_urbanizacao': taxa
                }
            except Exception:
                continue
                
        return {'territorio': nome_territorio, 'pop_total': 0, 'pop_urbana': 0, 'taxa_urbanizacao': 0}

ibge = ExtratorIBGE()
POP_BRASIL = ibge.obter_estimativa_populacao_atual()
meses_no_ano = 12

valor_dolar = 5.1 if len(sys.argv) <= 1 else float(sys.argv[1])

result_lines = []
agencias = pd.read_excel(cenarios_excel_path, sheet_name='agencias')
deploys = pd.read_excel(cenarios_excel_path, sheet_name='deploy')

agencias["duracao_norm"] = agencias["Duração"].str.lower()
agencias["duracao_norm"] = agencias["duracao_norm"].apply(lambda x: x.split(" ")[0])
agencias["territorios_lista"] = agencias['Território'].apply(lambda x: x.split(", "))

agencias_brasil = []

mensal_names = set(['mensal', 'mes'])
anual_names = set(['anual', 'ano'])

mensal_agencias = agencias[agencias['duracao_norm'].isin(mensal_names)]
soma_pop_agencias_mensal = mensal_agencias['População da Região'].sum()
soma_horas_mensais = mensal_agencias['Horas de Áudio Totais'].sum()

anual_agencias = agencias[agencias['duracao_norm'].isin(anual_names)]
if len(anual_agencias) > 0:
    soma_pop_agencias_anual = anual_agencias['População da Região'].sum()
    soma_horas_anuais = anual_agencias['Horas de Áudio Totais'].sum()
else:
    soma_pop_agencias_anual = soma_pop_agencias_mensal
    soma_horas_anuais = soma_horas_mensais * meses_no_ano

fator_escalamento_mensal = POP_BRASIL / soma_pop_agencias_mensal
fator_escalamento_anual = POP_BRASIL / soma_pop_agencias_anual

agencia_brasil_pop = {
    'Território': 'Brasil',
    'Horas de Áudio Totais': None,
    'Duração': 'Mensal',
    'População da Região': POP_BRASIL,
    'territorios_lista': ['Brasil']
}

agencia_brasil_anual = {
    'Território': 'Brasil',
    'Horas de Áudio Totais': None,
    'Duração': 'Anual',
    'População da Região': POP_BRASIL,
    'territorios_lista': ['Brasil']
}

agencias_brasil.append(agencia_brasil_pop)
agencias_brasil.append(agencia_brasil_anual)

agencias_brasil_df = pd.DataFrame(agencias_brasil)
agencias = pd.concat([agencias, agencias_brasil_df])

def calcular_demografia(territorios_lista):
    print(f"Consultando {territorios_lista}")
    pop_total_censo = 0
    pop_urbana_censo = 0
    
    for t in territorios_lista:
        print(f"  -> Consultando {t}")
        dados_ibge = ibge.obter_taxa_urbanizacao(t)
        if dados_ibge:
            pop_total_censo += dados_ibge['pop_total']
            pop_urbana_censo += dados_ibge['pop_urbana']
            print(f"    -> {dados_ibge}")
            
    taxa_urb = (pop_urbana_censo / pop_total_censo) if pop_total_censo > 0 else 0
    print(f"  -> Taxa de urbanização: {taxa_urb}")
    
    # Se o cenário for Brasil, calibramos os números para refletir a Estimativa Recente 
    # ao invés do número estático do Censo do ano anterior.
    if len(territorios_lista) == 1 and territorios_lista[0].lower() == 'brasil':
        pop_estimada_brasil = POP_BRASIL 
        pop_urbana_estimada = int(pop_estimada_brasil * taxa_urb)
        print(f"  -> Aplicando taxa estrutural ({taxa_urb:.2%}) sobre a estimativa recente ({pop_estimada_brasil}).")
        return pd.Series([pop_estimada_brasil, pop_urbana_estimada, taxa_urb])
    
    return pd.Series([pop_total_censo, pop_urbana_censo, taxa_urb])

# Aplica a função no dataframe, criando três novas colunas de uma só vez
print("Buscando dados demográficos na API do IBGE...")
agencias[['Pop_Total_IBGE', 'Pop_Urbana_IBGE', 'Taxa_Urbanizacao_IBGE']] = agencias['territorios_lista'].apply(calcular_demografia)

parsed_path = cenarios_excel_path.replace(".xlsx", "_agencias_parsed.csv")
print(f"Salvando dados demográficos em {parsed_path}")
agencias.to_csv(parsed_path, index=False)

pop_file = os.path.join(train_data_dir, 'populacao_brasil.txt')
with open(pop_file, 'w') as f:
    f.write(str(POP_BRASIL))
print(f"Salvando população do Brasil em {pop_file}")