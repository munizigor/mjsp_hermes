
var map;
var marker;

var latDefault = -15.7982; 
var lonDefault = -47.8755;

// Coordenadas Aracaju
// var defaultLat = -10.9095; 
// var defaultLon = -37.0748;

function initMap() {
    console.log("Inicializando o MAPA")
    const mapContainer = document.getElementById('map');
    if (!mapContainer) return;
    
    if (map) {
        map.remove();
        map = null;
    }

    const latStorage = sessionStorage.getItem('config_lat');
    const lonStorage = sessionStorage.getItem('config_lon');

    const initialLat = latStorage ? parseFloat(latStorage) : latDefault;
    const initialLon = lonStorage ? parseFloat(lonStorage) : lonDefault;

    map = L.map('map').setView([initialLat, initialLon], 15);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '© CARTO'
    }).addTo(map);

    marker = L.marker([initialLat, initialLon]).addTo(map)
        .bindPopup(latStorage ? 'Localização atualizada' : 'Aguardando localização...');

    setTimeout(() => { map.invalidateSize(); }, 200);
}

function atualizarMapa(lat, lon) {
    const novasCoords = [lat, lon];
    
    if (marker) {
        marker.setLatLng(novasCoords) 
              .setPopupContent('Localização Atualizada')
              .openPopup();
        
        map.setView(novasCoords, 15);
    }
}

window.addEventListener('novaLocalizacao', function(e) {
    atualizarMapa(e.detail.lat, e.detail.lon);
});

document.addEventListener('DOMContentLoaded', initMap);