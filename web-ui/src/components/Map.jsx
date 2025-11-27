
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, ZoomControl } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect } from 'react'

// Fix for default marker icon in leaflet with webpack/vite
import L from 'leaflet'
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow
});
L.Marker.prototype.options.icon = DefaultIcon;

function getColor(demand) {
    if (demand > 50) return '#ef4444' // red-500 (High)
    if (demand > 20) return '#f59e0b' // amber-500 (Medium)
    return '#10b981' // emerald-500 (Low)
}

function getRadius(demand) {
    // Reduced size range from 6-16 to 3-9 to reduce clutter
    return Math.max(3, Math.min(9, 2 + Math.sqrt(demand)))
}

export function Map({ stations, onStationClick }) {
    const torontoCoords = [43.6532, -79.3832]

    return (
        <MapContainer
            center={torontoCoords}
            zoom={13}
            scrollWheelZoom={true}
            className="w-full h-full"
            zoomControl={false}
        >
            <ZoomControl position="bottomright" />
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />

            {stations.map(station => (
                <CircleMarker
                    key={station.station_id}
                    center={[station.lat, station.lon]}
                    radius={getRadius(station.predicted_demand)}
                    pathOptions={{
                        color: 'white',
                        weight: 0.5, // Thinner border for smaller dots
                        fillColor: getColor(station.predicted_demand),
                        fillOpacity: 0.9 // Higher opacity to make them pop
                    }}
                    eventHandlers={{
                        click: () => onStationClick(station),
                    }}
                >
                    <Popup>
                        <div className="text-sm">
                            <p className="font-bold">{station.name}</p>
                            <p>Predicted Demand: {station.predicted_demand.toFixed(1)} trips</p>
                        </div>
                    </Popup>
                </CircleMarker>
            ))}
        </MapContainer>
    )
}
