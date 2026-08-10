import json
import requests
import time
import os
import sys
import signal

# Variável global para sinalizar interrupção
stop_threads = False


def signal_handler(sig, frame):
    """Captura o sinal de interrupção (CTRL+C) e define o sinalizador global."""
    global stop_threads
    print("\nInterrupção detectada! Finalizando threads...")
    stop_threads = True


# Registrar o manipulador de sinal para SIGINT
signal.signal(signal.SIGINT, signal_handler)


def start_emergency(hermes_url, hermes_headers, operator_code):
    """Inicia uma nova emergência na API do Hermes"""
    try:
        response = requests.get(
            f"{hermes_url}/start_call_interpretation/",
            params={"operator_unique_code": operator_code},
            headers=hermes_headers,
        )
        if response.status_code == 200:
            data = response.json()
            return data["new_emergency_id"]
        else:
            print(f"Erro ao iniciar emergência: {response.status_code}")
            return None
    except Exception as e:
        print(f"Erro ao conectar com API: {e}")
        return None


def send_audio(hermes_url, hermes_headers, emergency_id, audio_path, sample_rate=16000):
    """Envia áudio para a API do Hermes"""
    try:
        """with open(audio_path, 'rb') as audio_file:
        files = {'audio_data': audio_file}
        post_content = {
            'id_emergencia': str(emergency_id),
            'sp_rate': sample_rate
        }"""

        files = {"audio_data": open(audio_path, "rb")}
        response = requests.post(
            f"{hermes_url}/send_audio/?id_emergencia={emergency_id}&sp_rate={sample_rate}",
            files=files,
            verify=False,
            headers=hermes_headers,
        )

        # response = requests.post(f"{hermes_url}/send_audio/",
        #                        files=files, data=data)
        files["audio_data"].close()
        if response.status_code == 200:
            print(f"Áudio enviado com sucesso para emergência {emergency_id}")
            return True
        else:
            print(f"Erro ao enviar áudio: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erro ao enviar áudio: {e}")
        return False


def end_emergency(hermes_url, hermes_headers, emergency_id):
    """Encerra uma emergência na API do Hermes"""
    try:
        response = requests.post(
            f"{hermes_url}/end_call/",
            params={"id_emergencia": emergency_id},
            headers=hermes_headers,
        )
        if response.status_code == 200:
            print(f"Emergência {emergency_id} encerrada com sucesso")
            return True
        else:
            print(f"Erro ao encerrar emergência: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erro ao encerrar emergência: {e}")
        return False


def process_audio_call(hermes_url, audio_info, audio_path, operator_code):
    global stop_threads
    """Processa uma chamada de áudio completa"""
    try:
        # Delay para respeitar a carga
        delay = audio_info.get("Delay", 0)
        if delay > 0:
            print(f"Áudio {audio_info['ID']} aguardando {delay:.1f}s para começar")
            time.sleep(delay)

        if stop_threads:
            return

        call_start = time.time()
        # Inicia a emergência
        print(f"Iniciando emergência para áudio {audio_info['ID']}")
        emergency_id = start_emergency(hermes_url, operator_code)

        if emergency_id is None:
            print(f"Falha ao iniciar emergência para áudio {audio_info['ID']}")
            return

        print(f"Emergência {emergency_id} iniciada para áudio {audio_info['ID']}")

        # Envia o áudio
        print(f"Enviando áudio para emergência {emergency_id}")
        inf_results = {}
        for audio_part in audio_info["Segmentos"]:
            send_start = time.time()
            audio_path = audio_part["path"]
            duracao_real = audio_part["Duração Real"]
            if stop_threads:
                return
            print(f"Enviando parte do áudio: {audio_part}")
            audio_sent = send_audio(hermes_url, emergency_id, audio_path)
            if not audio_sent:
                print(f"Falha ao enviar áudio para emergência {emergency_id}")
                return
            send_audio_duration = time.time() - send_start
            to_sleep = duracao_real - send_audio_duration
            if to_sleep > 0:
                print(f"Aguardando {to_sleep:.1f} segundos para próxima parte do áudio")
                time.sleep(to_sleep)
            try:
                inf_results = requests.get(
                    f"{hermes_url}/get_all_inference_results/",
                    params={"id_emergencia": emergency_id},
                ).json()
                # print(inf_results)
            except Exception as err:
                print("Erro tentando obter resultados:", err, err.__context__)
                pass

        if stop_threads:
            return

        """duration_seconds = audio_info['Duracao Real']
        time_passed = time.time() - call_start
        to_sleep = duration_seconds - time_passed
        print(f"Aguardando {to_sleep:.1f} segundos (tempo restante da ligação)")
        time.sleep(to_sleep + 1)"""

        # Encerra a emergência
        print(f"Encerrando emergência {emergency_id}")
        end_emergency(hermes_url, emergency_id)

        houve_delay = False
        delay_local = 0.0
        delay_wait_time = 1
        max_wait_time = 180  # Tempo máximo de espera em segundos
        wait_start = time.time()
        while True:
            if time.time() - wait_start > max_wait_time:
                print(f"Tempo máximo de espera excedido para emergência {emergency_id}")
                break
            end_status = requests.get(
                f"{hermes_url}/processing_finished/",
                params={"id_emergencia": emergency_id},
            )
            content = end_status.json()
            if content["terminado"] == False:
                print(emergency_id, content)
                houve_delay = True
                time.sleep(delay_wait_time)
            else:
                delay_local = time.time() - wait_start
                break

            try:
                inf_results = requests.get(
                    f"{hermes_url}/get_all_inference_results/",
                    params={"id_emergencia": emergency_id},
                ).json()
            except Exception as err:
                print("Erro tentando obter resultados:", err, err.__context__)
                pass

        if delay_local <= 2:
            delay_local = 0.0

        print(f"Chamada {audio_info['ID']} processada com sucesso")

        try:
            inf_results = requests.get(
                f"{hermes_url}/get_all_inference_results/",
                params={"id_emergencia": emergency_id},
            ).json()
            print(inf_results)
        except Exception as err:
            print("Erro tentando obter resultados:", err, err.__context__)
            pass

        final_result = {
            "dataset_index": audio_info["ID"],
            "id_emergencia": emergency_id,
            "delay_local": delay_local,
            "results": inf_results,
        }

        filename = "run_tests-result." + str(audio_info["ID"]) + ".json"
        fh = open(filename, "w")
        json.dump(final_result, fh, ensure_ascii=False)
        fh.close()

    except Exception as e:
        print(f"Erro ao processar chamada {audio_info['ID']}: {e}", file=sys.stderr)
        print(e, file=sys.stderr)
        raise (e)


def wait_for_final_results(emergency_id, hermes_url, hermes_headers):
    inf_results = None
    houve_delay = False
    delay_local = 0.0
    delay_wait_time = 1
    max_wait_time = 180  # Tempo máximo de espera em segundos
    wait_start = time.time()
    while True:
        if time.time() - wait_start > max_wait_time:
            print(f"Tempo máximo de espera excedido para emergência {emergency_id}")
            break
        end_status = requests.get(
            f"{hermes_url}/processing_finished/",
            params={"id_emergencia": emergency_id},
            headers=hermes_headers,
        )
        content = end_status.json()
        if not content["terminado"]:
            print(emergency_id, "atrasado a", time.time() - wait_start)
            # print(emergency_id, content)
            houve_delay = True
            time.sleep(delay_wait_time)
        else:
            delay_local = time.time() - wait_start
            break

        try:
            inf_results = requests.get(
                f"{hermes_url}/get_all_inference_results/",
                params={"id_emergencia": emergency_id},
                headers=hermes_headers,
            ).json()
        except Exception as err:
            print("Erro tentando obter resultados:", err, err.__context__)
            pass

    if delay_local <= 2:
        delay_local = 0.0

    return delay_local, houve_delay, inf_results


def process_audio_call_hf(
    hermes_url, hermes_headers, audio_info, operator_code, output_dir
):
    global stop_threads
    """Processa uma chamada de áudio completa"""
    try:
        # Delay para respeitar a carga
        delay = audio_info.get("delay", 0)
        if delay > 0:
            print(f"Áudio {audio_info['ID']} aguardando {delay:.1f}s para começar")
            time.sleep(delay)
        else:
            print(f"Áudio {audio_info['ID']} iniciando imediatamente")

        if stop_threads:
            return

        call_start = time.time()
        # Inicia a emergência
        print(f"Iniciando emergência para áudio {audio_info['ID']}")
        emergency_id = start_emergency(hermes_url, hermes_headers, operator_code)

        if emergency_id is None:
            print(f"Falha ao iniciar emergência para áudio {audio_info['ID']}")
            return

        print(f"Emergência {emergency_id} iniciada para áudio {audio_info['ID']}")

        # Envia o áudio
        print(f"Enviando áudio para emergência {emergency_id}")
        inf_results = {}
        audio_paths = audio_info["audio_paths"]
        audio_lengths = list(audio_info["audio_lengths"])

        lens_sum = sum(audio_lengths)
        current_sum = 0.0
        send_audio_duration = 0.0
        inf_get_duration = 0.0
        last_length = 0.0
        for audio_path, audio_length in zip(audio_paths, audio_lengths):
            # send_start = time.time()
            if stop_threads:
                return
            to_sleep = audio_length - send_audio_duration - inf_get_duration
            if to_sleep > 0:
                print(f"{emergency_id} esperando {to_sleep}s")
                time.sleep(to_sleep)
            send_start = time.time()
            audio_sent = send_audio(
                hermes_url, hermes_headers, emergency_id, audio_path
            )
            if not audio_sent:
                print(f"Falha ao enviar áudio para emergência {emergency_id}")
                return
            current_sum += audio_length
            print(f"{emergency_id} em {(current_sum/lens_sum)*100}%")
            send_audio_duration = time.time() - send_start
            inf_get_start = time.time()
            try:
                inf_results = requests.get(
                    f"{hermes_url}/get_all_inference_results/",
                    params={"id_emergencia": emergency_id},
                    headers=hermes_headers,
                ).json()
                # print(inf_results)
            except Exception as err:
                print("Erro tentando obter resultados:", err, err.__context__)
                pass
            # Remove audio_path
            os.remove(audio_path)
            inf_get_duration = time.time() - inf_get_start
            last_length = audio_length

        if stop_threads:
            return

        """duration_seconds = audio_info['Duracao Real']
        time_passed = time.time() - call_start
        to_sleep = duration_seconds - time_passed
        print(f"Aguardando {to_sleep:.1f} segundos (tempo restante da ligação)")
        time.sleep(to_sleep + 1)"""

        # Encerra a emergência
        print(f"Encerrando emergência {emergency_id}")
        end_emergency(hermes_url, hermes_headers, emergency_id)

        to_sleep = last_length - send_audio_duration - inf_get_duration
        if to_sleep > 0:
            print(f"Aguardando {to_sleep:.1f} segundos (duração do ultimo audio)")
            time.sleep(to_sleep)

        delay_local, houve_delay, inf_results_from_wait = wait_for_final_results(
            emergency_id, hermes_url, hermes_headers
        )

        if inf_results_from_wait is not None:
            inf_results = inf_results_from_wait

        print(f"Chamada {audio_info['ID']} processada com sucesso")

        try:
            inf_results = requests.get(
                f"{hermes_url}/get_all_inference_results/",
                params={"id_emergencia": emergency_id},
                headers=hermes_headers,
            ).json()
            print(inf_results)
        except Exception as err:
            print("Erro tentando obter resultados:", err, err.__context__)
            pass

        final_result = {
            "dataset_index": audio_info["ID"],
            "id_emergencia": emergency_id,
            "call_start_delay": delay,
            "audio_lengths": audio_lengths,
            "delay_local": delay_local,
            "results": inf_results,
        }

        filename = output_dir + "/run_tests-result." + str(audio_info["ID"]) + ".json"
        fh = open(filename, "w")
        json.dump(final_result, fh, ensure_ascii=False)
        fh.close()
    except Exception as e:
        print(f"Erro ao processar chamada {audio_info['ID']}: {e}", file=sys.stderr)
        print(e, file=sys.stderr)
        raise (e)
