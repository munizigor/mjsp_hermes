#!/usr/bin/env python3

"""
Michell - 2025-10-01

Direct Monitor - Monitoramento e processamento direto de arquivos WAV em crescimento

Este script monitora o diretório de gravações do Asterisk e processa diretamente
os arquivos WAV em crescimento, ignorando o cabeçalho WAV e tratando os arquivos
como fluxos contínuos de dados brutos.

Uso:
    python direct_monitor.py /caminho/para/diretorio/monitor [diretório_saída]
"""

import os
import sys
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Importar o processador direto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from direct_raw_processor import DirectRawProcessor

class DirectRecordingHandler(FileSystemEventHandler):
    """
    Manipulador de eventos para monitorar novas gravações do Asterisk.
    """
    
    def __init__(self, output_dir, folga_db=12, min_silence_len=2000, keep_silence=250):
        """
        Inicializa o manipulador de eventos.
        
        Args:
            output_dir (str): Diretório para salvar os segmentos
            folga_db (float): Valor em dB para subtrair do volume médio
            min_silence_len (int): Duração mínima do silêncio em ms
            keep_silence (int): Quantidade de silêncio a manter no início e fim dos segmentos em ms
        """
        self.output_dir = output_dir
        self.folga_db = folga_db
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.active_processors = {}
    
    def on_created(self, event):
        """
        Chamado quando um novo arquivo é criado no diretório monitorado.
        
        Args:
            event: Evento de criação de arquivo
        """
        if not event.is_directory and event.src_path.endswith('.wav'):
            print(f"Nova gravação detectada: {event.src_path}")
            
            # Iniciar um novo processo para processar este arquivo
            self._start_processor(event.src_path)
    
    def _start_processor(self, wav_file):
        """
        Inicia um novo processador para o arquivo de áudio.
        
        Args:
            wav_file (str): Caminho para o arquivo WAV
        """
        try:
            # Criar um diretório específico para os segmentos deste arquivo
            base_name = os.path.basename(wav_file).replace('.wav', '')
            segments_dir = os.path.join(self.output_dir, base_name)
            os.makedirs(segments_dir, exist_ok=True)
            
            # Iniciar um processo para processar diretamente o arquivo
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'direct_raw_processor.py'),
                wav_file,
                segments_dir,
                '--folga-db', str(self.folga_db),
                '--min-silence', str(self.min_silence_len),
                '--keep-silence', str(self.keep_silence)
            ]
            
            process = subprocess.Popen(cmd)
            print(f"Processamento direto iniciado para {wav_file} (PID: {process.pid})")
            
            # Armazenar o processo para referência futura
            self.active_processors[wav_file] = process
        
        except Exception as e:
            print(f"Erro ao iniciar processador para {wav_file}: {e}")
    
    def process_existing_files(self, monitor_dir):
        """
        Processa arquivos WAV existentes no diretório monitorado.
        
        Args:
            monitor_dir (str): Diretório a ser verificado
        """
        for filename in os.listdir(monitor_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(monitor_dir, filename)
                if os.path.isfile(file_path):
                    print(f"Processando arquivo existente: {file_path}")
                    self._start_processor(file_path)
    
    def cleanup(self):
        """
        Limpa os recursos utilizados pelo manipulador de eventos.
        """
        # Encerrar todos os processos ativos
        for wav_file, process in self.active_processors.items():
            try:
                process.terminate()
                print(f"Processo para {wav_file} encerrado.")
            except:
                pass

def main():
    """
    Função principal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitora e processa diretamente arquivos WAV em crescimento.')
    parser.add_argument('monitor_dir', help='Diretório de gravações do Asterisk a ser monitorado')
    parser.add_argument('--output-dir', '-o', help='Diretório para salvar os segmentos')
    parser.add_argument('--folga-db', '-f', type=float, default=12.0, 
                        help='Valor em dB para subtrair do volume médio (padrão: 12)')
    parser.add_argument('--min-silence', '-m', type=int, default=1000, 
                        help='Duração mínima do silêncio em ms (padrão: 1000)')
    parser.add_argument('--keep-silence', '-k', type=int, default=250, 
                        help='Quantidade de silêncio a manter no início e fim dos segmentos em ms (padrão: 250)')
    
    args = parser.parse_args()
    
    # Se não foi especificado um diretório de saída, usar um subdiretório 'segments' no diretório monitorado
    output_dir = args.output_dir or os.path.join(args.monitor_dir, 'segments')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Monitorando o diretório {args.monitor_dir} para novas gravações...")
    print(f"Segmentos serão salvos em {output_dir}")
    print(f"Configuração: folga_db={args.folga_db}, min_silence={args.min_silence}ms, keep_silence={args.keep_silence}ms")
    
    # Criar o manipulador de eventos
    event_handler = DirectRecordingHandler(
        output_dir=output_dir,
        folga_db=args.folga_db,
        min_silence_len=args.min_silence,
        keep_silence=args.keep_silence
    )
    
    # Processar arquivos existentes
    print("Verificando arquivos existentes...")
    event_handler.process_existing_files(args.monitor_dir)
    
    # Configurar o observador
    observer = Observer()
    observer.schedule(event_handler, args.monitor_dir, recursive=True)
    
    # Iniciar o monitoramento
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido pelo usuário.")
        event_handler.cleanup()
        observer.stop()
    except Exception as e:
        print(f"Erro durante o monitoramento: {e}")
        event_handler.cleanup()
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
