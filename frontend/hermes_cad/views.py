import logging
import os
import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.test import RequestFactory
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

HERMES_BACKEND_URL = settings.HERMES_BACKEND_URL
HERMES_API_KEY = settings.HERMES_API_KEY

class GeoService:
    def __init__(self, contact_email="rafaelao.marques@gmail.com"):

        self.geolocator = Nominatim(user_agent=f"Hermes_v1({contact_email})")

        self.geocode = RateLimiter(
            self.geolocator.geocode, 
            min_delay_seconds=1, 
            max_retries=2
        )

    def get_coordinates(self, address):
        if not address:
            return None, None
        try:
            location = self.geocode(address)
            if location:
                return location.latitude, location.longitude
            logger.warning(f"Endereço não localizado: {address}")
        except Exception as e:
            logger.error(f"Falha crítica no serviço de geolocalização: {e}")
        return None, None

geo_service = GeoService()

def get_hermes_auth_headers():
    return {
        "X-Hermes-API-Key": HERMES_API_KEY,
        "Accept": "application/json"
    }

def dashboard(request):
    return render(request, 'hermes_cad/dashboard.html')

def home(request):
    return render(request, 'hermes_cad/home.html')

def emergency_info(request):
    id_emergencia = request.GET.get('id_emergencia')
    if not id_emergencia:
        return JsonResponse({'error': 'id_emergencia is required'}, status=400)

    # --- Your existing Logic to fetch data ---
    factory = RequestFactory()
    dummy_request = factory.get(f'/proxy/get_all_inference_results/?id_emergencia={id_emergencia}')
    dummy_request.GET = dummy_request.GET.copy()
    dummy_request.GET['id_emergencia'] = id_emergencia

    response_from_proxy = proxy_get_all_inference_results(dummy_request)

    if response_from_proxy.status_code != 200:
        # Handle error gracefully for HTMX to avoid breaking the UI
        return JsonResponse({'error': 'Failed to retrieve details.'}, status=response_from_proxy.status_code)

    try:
        emergency_data = json.loads(response_from_proxy.content)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON")
        return JsonResponse({'error': 'Invalid JSON.'}, status=500)
    # -----------------------------------------

    context = {
        'id_emergencia': id_emergencia, 
        'emergency_data': emergency_data
    }

    # CHECK FOR HTMX HEADER
    if request.headers.get('HX-Request'):
        # If HTMX, render ONLY the partial grid
        return render(request, 'partials/emergency_content.html', context)

    # If normal load, render the full page (which includes the partial)
    return render(request, 'hermes_cad/emergency_info.html', context)

def emergencias(request):

    try:
        response = requests.get(
            f"{HERMES_BACKEND_URL}/list_emergencies/",
            headers=get_hermes_auth_headers(),
            timeout=20
        )
        response.raise_for_status()
        # Attempt to parse as JSON, if it fails, return text response
        try:
            response_from_proxy = response.json()
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON")
            return JsonResponse({'error': 'Invalid JSON.', 'message': response.text}, status=500)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

    # -----------------------------------------

    context = {
        'emergency_list': response_from_proxy['emergencies']
    }

    # CHECK FOR HTMX HEADER
    if request.headers.get('HX-Request'):
        # If HTMX, render ONLY the partial grid
        return render(request, 'partials/emergency_call.html', context)

    # If normal load, render the full page (which includes the partial)
    return render(request, 'hermes_cad/emergencias.html', context)

def proxy_emergencias(request):
    try:
        response = requests.get(
            f"{HERMES_BACKEND_URL}/list_emergencies/",
            headers=get_hermes_auth_headers(),
            timeout=5
        )
        response.raise_for_status()
        try:
            response_from_proxy = response.json()
        except json.JSONDecodeError:
            logger.error(f"Erro ao decodificar JSON do Hermes Backend")
            return JsonResponse({'error': 'Resposta do servidor não é um JSON válido.'}, status=500)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro de conexão no proxy_emergencias: {str(e)}")
        # Retornamos uma mensagem genérica para o front por segurança
        return JsonResponse({'error': 'Não foi possível buscar as emergências agora.'}, status=502)

    if isinstance(response_from_proxy, dict):
        emergencies = response_from_proxy.get('emergencies', [])
    else:
        emergencies = response_from_proxy if isinstance(response_from_proxy, list) else []

    return JsonResponse({'calls': emergencies})

def proxy_start_call_interpretation(request):
    operator_unique_code = request.GET.get('operator_unique_code')
    if not operator_unique_code:
        return JsonResponse({'error': 'operator_unique_code is required'}, status=400)
    
    logger.info(f"🌍 HERMES_BACKEND_URL carregado do settings: {settings.HERMES_BACKEND_URL}")

    try:
        response = requests.get(
            f"{HERMES_BACKEND_URL}/start_call_interpretation/",
            params={'operator_unique_code': operator_unique_code},
            headers=get_hermes_auth_headers(),
            timeout=20
        )
        response.raise_for_status()
        return JsonResponse(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with backend: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def proxy_end_call(request):
    id_emergencia = request.GET.get('id_emergencia')
    if not id_emergencia:
        return JsonResponse({'error': 'id_emergencia is required'}, status=400)
    
    try:
        response = requests.post(
            f"{HERMES_BACKEND_URL}/end_call/", 
            params={'id_emergencia': id_emergencia},
            headers=get_hermes_auth_headers(), 
            timeout=20
        )
        response.raise_for_status()
        # Attempt to parse as JSON, if it fails, return text response
        try:
            return JsonResponse(response.json(), safe=False)
        except json.JSONDecodeError:
            return JsonResponse({'message': response.text}, status=response.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)

def proxy_send_transcript(request):
    if request.method == 'POST':
        id_emergencia = request.GET.get('id_emergencia')
        transcript = request.GET.get('transcript') # Changed to GET since the frontend uses data-urlencode on GET
        
        if not id_emergencia or not transcript:
            return JsonResponse({'error': 'id_emergencia and transcript are required'}, status=400)
        
        try:
            # The original JS uses GET parameters even for a POST request (curl -G)
            # So, we'll send it as a GET request to the backend with parameters
            # and set the method to POST on the backend, or adapt the backend call.
            # For now, let's assume the backend expects POST with query params.
            response = requests.post(
                f"{HERMES_BACKEND_URL}/send_transcript/", 
                params={'id_emergencia': id_emergencia, 'transcript': transcript},
                headers=get_hermes_auth_headers(),
                timeout=20
            )
            response.raise_for_status()
            return JsonResponse(response.json(), safe=False)
        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

def proxy_get_all_inference_results(request):
    id_emergencia = request.GET.get('id_emergencia')
    if not id_emergencia:
        return JsonResponse({'error': 'id_emergencia is required'}, status=400)
    
    try:
        response = requests.get(
            f"{HERMES_BACKEND_URL}/get_all_inference_results/", 
            params={'id_emergencia': id_emergencia},
            headers=get_hermes_auth_headers(),
            timeout=30
        )
        
        response.raise_for_status()
        
        data = response.json()

        address = data.get("rua_ou_logradouro", [])
        district = data.get("bairro", [])
        city = data.get("municipio", [])

        rua = address[0].get('value', '') if address else ""
        bairro = district[0].get('value', '') if district else ""
        cidade = city[0].get('value', '') if city else ""

        if len(rua) > 5:
            get_address = f"{rua}, {bairro}, {cidade}" 
            
            # TODO: pegar o operator_email    
            lat, lon = geo_service.get_coordinates(get_address)
            
            # Injetamos lat/lon no JSON original sem alterar a estrutura da inferência
            data['lat'] = lat
            data['lon'] = lon
        else:
            data['lat'] = None
            data['lon'] = None

        return JsonResponse(data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na chamada ao Hermes: {e}")
        return JsonResponse({'error': str(e)}, status=500)
