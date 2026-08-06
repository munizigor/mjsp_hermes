#!/usr/bin/env python3
"""
Michell - 2025-10-02

Transcribe Segments - Envia segmentos de áudio para o Azure OpenAI Whisper para transcrição

Este script monitora um diretório de segmentos de áudio e os envia para o Azure OpenAI Whisper
para transcrição, salvando os resultados em arquivos de texto.

Uso:
    python transcribe_segments.py /caminho/para/diretorio/segmentos [diretório_saída]

"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import AzureOpenAI

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transcribe_segments.log")
    ]
)
logger = logging.getLogger("TranscribeSegments")

class TranscriptionHandler(FileSystemEventHandler):
    """
    Manipulador de eventos para monitorar novos segmentos de áudio e transcrevê-los.
    """
    
    def __init__(self, output_dir, azure_client, deployment_id):
        """
        Inicializa o manipulador de eventos.
        
        Args:
            output_dir (str): Diretório para salvar as transcrições
            azure_client: Cliente do Azure OpenAI
            deployment_id (str): ID do deployment do modelo Whisper
        """
        self.output_dir = output_dir
        self.azure_client = azure_client
        self.deployment_id = deployment_id
        self.active_transcriptions = {}
        
        # Criar diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_created(self, event):
        """
        Chamado quando um novo arquivo é criado no diretório monitorado.
        
        Args:
            event: Evento de criação de arquivo
        """
        if not event.is_directory and event.src_path.endswith('.wav'):
            logger.info(f"Novo segmento de áudio detectado: {event.src_path}")
            
            # Iniciar transcrição para este arquivo
            self._transcribe_segment(event.src_path)
    
    def _transcribe_segment(self, audio_file):
        """
        Transcreve um segmento de áudio usando o Azure OpenAI Whisper.
        
        Args:
            audio_file (str): Caminho para o arquivo de áudio
        """
        try:
            # Verificar se o arquivo existe
            if not os.path.exists(audio_file):
                logger.error(f"Arquivo {audio_file} não encontrado.")
                return
            
            # Verificar se o arquivo está vazio ou muito pequeno
            if os.path.getsize(audio_file) < 1024:  # Menos de 1KB
                logger.warning(f"Arquivo {audio_file} muito pequeno, pulando.")
                return
            
            # Gerar nome do arquivo de saída
            base_name = os.path.basename(audio_file).replace('.wav', '')
            output_file = os.path.join(self.output_dir, f"{base_name}.txt")
            json_output_file = os.path.join(self.output_dir, f"{base_name}.json")
            
            # Verificar se já existe uma transcrição para este arquivo
            if os.path.exists(output_file):
                logger.info(f"Transcrição já existe para {audio_file}, pulando.")
                return
            
            logger.info(f"Iniciando transcrição para {audio_file}")
            
            # Enviar o arquivo para o Azure OpenAI Whisper
            with open(audio_file, "rb") as audio:
                result = self.azure_client.audio.transcriptions.create(
                    file=audio,
                    model=self.deployment_id
                )
            
            # Salvar o resultado como texto
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.text)
            
            # Salvar o resultado completo como JSON
            with open(json_output_file, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcrição concluída para {audio_file}")
            logger.info(f"Texto: {result.text[:100]}...")
            
        except Exception as e:
            logger.error(f"Erro ao transcrever {audio_file}: {e}")
    
    def process_existing_files(self, segments_dir):
        """
        Processa arquivos WAV existentes no diretório monitorado.
        
        Args:
            segments_dir (str): Diretório a ser verificado
        """
        for filename in os.listdir(segments_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(segments_dir, filename)
                if os.path.isfile(file_path):
                    logger.info(f"Processando arquivo existente: {file_path}")
                    self._transcribe_segment(file_path)

def main():
    """
    Função principal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitora e transcreve segmentos de áudio usando Azure OpenAI Whisper.')
    parser.add_argument('segments_dir', help='Diretório de segmentos de áudio a ser monitorado')
    parser.add_argument('--output-dir', '-o', help='Diretório para salvar as transcrições')
    parser.add_argument('--api-key', help='Chave da API do Azure OpenAI')
    parser.add_argument('--endpoint', help='Endpoint do Azure OpenAI')
    parser.add_argument('--deployment-id', help='ID do deployment do modelo Whisper')
    parser.add_argument('--api-version', default="2024-02-01", help='Versão da API do Azure OpenAI')
    
    args = parser.parse_args()
    
    # Se não foi especificado um diretório de saída, usar um subdiretório 'transcriptions' no diretório monitorado
    output_dir = args.output_dir or os.path.join(args.segments_dir, 'transcriptions')
    os.makedirs(output_dir, exist_ok=True)
    
    # Obter credenciais do Azure OpenAI
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = args.deployment_id or os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    api_version = args.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    if not api_key or not endpoint or not deployment_id:
        logger.error("É necessário fornecer api-key, endpoint e deployment-id.")
        logger.error("Você pode fornecê-los como argumentos ou como variáveis de ambiente:")
        logger.error("  AZURE_OPENAI_API_KEY")
        logger.error("  AZURE_OPENAI_ENDPOINT")
        logger.error("  AZURE_OPENAI_DEPLOYMENT_ID")
        sys.exit(1)
    
    # Inicializar cliente do Azure OpenAI
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        logger.info(f"Cliente do Azure OpenAI inicializado com sucesso. Endpoint: {endpoint}")
    except Exception as e:
        logger.error(f"Erro ao inicializar cliente do Azure OpenAI: {e}")
        sys.exit(1)
    
    logger.info(f"Monitorando o diretório {args.segments_dir} para novos segmentos de áudio...")
    logger.info(f"Transcrições serão salvas em {output_dir}")
    logger.info(f"Usando deployment_id: {deployment_id}")
    
    # Criar o manipulador de eventos
    event_handler = TranscriptionHandler(
        output_dir=output_dir,
        azure_client=client,
        deployment_id=deployment_id
    )
    
    # Processar arquivos existentes
    logger.info("Verificando arquivos existentes...")
    event_handler.process_existing_files(args.segments_dir)
    
    # Configurar o observador
    observer = Observer()
    observer.schedule(event_handler, args.segments_dir, recursive=False)
    
    # Iniciar o monitoramento
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nMonitoramento interrompido pelo usuário.")
        observer.stop()
    except Exception as e:
        logger.error(f"Erro durante o monitoramento: {e}")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
