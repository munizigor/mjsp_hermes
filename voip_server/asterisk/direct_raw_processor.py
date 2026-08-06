#!/usr/bin/env python3
"""
Michell - 2025-10-01

Direct Raw Processor - Processador direto de dados brutos de arquivos WAV em crescimento

Este script acessa diretamente os dados brutos de um arquivo WAV em crescimento,
ignorando o cabeçalho WAV e tratando o arquivo como um fluxo contínuo de dados.

Uso:
    python direct_raw_processor.py arquivo.wav [diretório_saída]

"""

import os
import sys
import time
import wave
import struct
import audioop
import numpy as np
from datetime import datetime
from collections import deque

class DirectRawProcessor:
    """
    Processador direto de dados brutos de arquivos WAV em crescimento.
    """
    
    def __init__(self, wav_file, output_dir=None, folga_db=12, min_silence_len=1000, 
                 keep_silence=250, sample_rate=8000, sample_width=2, channels=1):
        """
        Inicializa o processador de dados brutos.
        
        Args:
            wav_file (str): Caminho para o arquivo WAV em crescimento
            output_dir (str): Diretório para salvar os segmentos
            folga_db (float): Valor em dB para subtrair do volume médio para definir o limiar de silêncio
            min_silence_len (int): Duração mínima do silêncio em ms para considerar um ponto de corte
            keep_silence (int): Quantidade de silêncio a manter no início e fim dos segmentos em ms
            sample_rate (int): Taxa de amostragem em Hz (padrão: 8000)
            sample_width (int): Largura da amostra em bytes (padrão: 2 = 16 bits)
            channels (int): Número de canais (padrão: 1 = mono)
        """
        self.wav_file = wav_file
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(wav_file))
        self.folga_db = folga_db
        self.min_silence_len = min_silence_len
        self.keep_silence = keep_silence
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        
        # Posição atual no arquivo
        self.current_position = 0
        
        # Tamanho do cabeçalho WAV padrão (44 bytes)
        self.header_size = 44
        
        # Tamanho do chunk em bytes (100ms de áudio)
        self.chunk_size = int(self.sample_rate * self.sample_width * self.channels * 0.1)
        
        # Número de chunks para considerar silêncio
        self.min_silence_chunks = int(self.min_silence_len / 100)  # min_silence_len em ms / 100ms por chunk
        
        # Número de chunks de silêncio a manter
        self.keep_silence_chunks = int(self.keep_silence / 100)  # keep_silence em ms / 100ms por chunk
        
        # Buffer para armazenar os dados de áudio acumulados
        self.audio_buffer = []
        
        # Contador de segmentos
        self.segment_count = 0
        
        # Histórico de volume para cálculo dinâmico do limiar de silêncio
        self.volume_history = deque(maxlen=50)  # Manter apenas os últimos 50 chunks
        
        # Limiar de silêncio dinâmico
        self.silence_threshold = None
        
        # Verificar se o arquivo existe ou esperar por ele
        self._wait_for_file()
        
        # Criar o diretório de saída se não existir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Processador direto iniciado para {wav_file}")
        print(f"Segmentos serão salvos em {self.output_dir}")
        print(f"Configuração: folga_db={folga_db}, min_silence_len={min_silence_len}ms, keep_silence={keep_silence}ms")
        print(f"Formato de áudio: {channels} canais, {sample_width*8} bits, {sample_rate} Hz")
    
    def _wait_for_file(self):
        """Espera até que o arquivo exista."""
        while not os.path.exists(self.wav_file):
            print(f"Aguardando a criação do arquivo {self.wav_file}...")
            time.sleep(1)
    
    def _read_new_data(self):
        """Lê novos dados do arquivo em crescimento."""
        try:
            with open(self.wav_file, 'rb') as f:
                # Ir para a posição atual
                f.seek(self.current_position)
                
                # Ler um chunk de dados
                chunk = f.read(self.chunk_size)
                
                # Atualizar a posição atual
                self.current_position = f.tell()
                
                return chunk
        except Exception as e:
            print(f"Erro ao ler novos dados: {e}")
            return b''
    
    def _calculate_silence_threshold(self):
        """Calcula o limiar de silêncio dinâmico com base no volume médio."""
        if not self.volume_history:
            return 500  # Valor padrão se não houver histórico
        
        # Calcular o RMS médio
        avg_rms = sum(self.volume_history) / len(self.volume_history)
        
        # Converter para dB
        if avg_rms > 0:
            avg_db = 20 * np.log10(avg_rms)
            
            # Aplicar a folga e converter de volta para RMS
            threshold_db = avg_db - self.folga_db
            threshold_rms = 10 ** (threshold_db / 20)
            
            print(f"Volume médio: {avg_db:.2f} dB")
            print(f"Limiar de silêncio: {threshold_db:.2f} dB ({threshold_rms:.2f} RMS)")
            
            return threshold_rms
        else:
            return 500  # Valor padrão se o RMS médio for zero
    
    def _is_silence(self, chunk):
        """Verifica se um chunk de áudio é silêncio."""
        if not chunk:
            return True
        
        # Calcular o RMS do chunk
        rms = audioop.rms(chunk, self.sample_width)
        
        # Adicionar ao histórico de volume
        self.volume_history.append(rms)
        
        # Recalcular o limiar de silêncio a cada 20 chunks
        if len(self.volume_history) >= 20 and len(self.volume_history) % 20 == 0:
            self.silence_threshold = self._calculate_silence_threshold()
        
        # Se ainda não temos um limiar, usar o valor padrão
        if self.silence_threshold is None:
            self.silence_threshold = 500
        
        return rms < self.silence_threshold
    
    def _save_segment(self, segment):
        """Salva um segmento de áudio como arquivo WAV."""
        if not segment:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        segment_filename = f"segment_{timestamp}_{self.segment_count}.wav"
        segment_path = os.path.join(self.output_dir, segment_filename)
        
        # Criar arquivo WAV
        with wave.open(segment_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(segment))
        
        print(f"Segmento salvo: {segment_path}")
        self.segment_count += 1
        
        return segment_path
    
    def process(self):
        """
        Processa o arquivo em crescimento e o segmenta em pontos de silêncio.
        """
        try:
            # Pular o cabeçalho WAV se o arquivo for grande o suficiente
            if os.path.getsize(self.wav_file) > self.header_size:
                self.current_position = self.header_size
                print(f"Pulando cabeçalho WAV ({self.header_size} bytes)")
            
            current_segment = []
            silence_count = 0
            is_in_silence = False
            
            while True:
                # Verificar se o arquivo ainda existe
                if not os.path.exists(self.wav_file):
                    print(f"Arquivo {self.wav_file} não encontrado. Aguardando...")
                    time.sleep(1)
                    continue
                
                # Ler novos dados
                chunk = self._read_new_data()
                
                # Se não há novos dados, esperar
                if not chunk:
                    time.sleep(0.1)
                    continue
                
                # Adicionar o chunk ao buffer de áudio
                self.audio_buffer.append(chunk)
                
                # Verificar se este chunk é silêncio
                if self._is_silence(chunk):
                    silence_count += 1
                    
                    # Se estamos em um segmento e encontramos silêncio suficiente, finalizar o segmento
                    if not is_in_silence and silence_count >= self.min_silence_chunks:
                        is_in_silence = True
                        
                        # Adicionar alguns chunks de silêncio ao final do segmento
                        silence_to_keep = min(silence_count, self.keep_silence_chunks)
                        current_segment.extend(self.audio_buffer[-silence_to_keep:])
                        
                        # Salvar o segmento se tiver conteúdo suficiente
                        if len(current_segment) > self.keep_silence_chunks * 2:
                            self._save_segment(current_segment)
                        
                        # Iniciar um novo segmento
                        current_segment = []
                else:
                    # Se estávamos em silêncio e agora temos áudio, iniciar um novo segmento
                    if is_in_silence:
                        is_in_silence = False
                        
                        # Adicionar alguns chunks de silêncio ao início do novo segmento
                        silence_to_keep = min(silence_count, self.keep_silence_chunks)
                        current_segment.extend(self.audio_buffer[-(silence_count+1):-silence_count+silence_to_keep-1])
                    
                    # Resetar o contador de silêncio
                    silence_count = 0
                    
                    # Adicionar este chunk ao segmento atual
                    current_segment.append(chunk)
                
                # Limitar o tamanho do buffer para economizar memória
                if len(self.audio_buffer) > 1000:  # Manter apenas os últimos 100 segundos
                    self.audio_buffer = self.audio_buffer[-1000:]
        
        except KeyboardInterrupt:
            print("\nProcessamento interrompido pelo usuário.")
        except Exception as e:
            print(f"Erro durante o processamento: {e}")
        finally:
            # Salvar o último segmento se tiver conteúdo
            if current_segment and len(current_segment) > self.keep_silence_chunks * 2:
                self._save_segment(current_segment)

def main():
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} arquivo.wav [diretório_saída]")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Iniciar o processador
    processor = DirectRawProcessor(wav_file, output_dir)
    processor.process()

if __name__ == "__main__":
    main()
