import { useState, useEffect } from 'react'
import axios from 'axios'
import { Map } from './components/Map'
import { StationPanel } from './components/StationPanel'
import { Bike } from 'lucide-react'

function App() {
  const [stations, setStations] = useState([])
  const [selectedStation, setSelectedStation] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch stations from backend
    axios.get('http://127.0.0.1:8000/stations')
      .then(res => {
        setStations(res.data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Failed to fetch stations", err)
        setLoading(false)
      })
  }, [])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-50 text-gray-900">
      {/* Sidebar / Panel */}
      <div className={`fixed inset-y-0 left-0 z-20 w-96 transform bg-white shadow-xl transition-transform duration-300 ease-in-out ${selectedStation ? 'translate-x-0' : '-translate-x-full'}`}>
        {selectedStation && (
          <StationPanel
            station={selectedStation}
            onClose={() => setSelectedStation(null)}
          />
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 relative">
        {/* Header Overlay */}
        <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-gray-200 max-w-sm">
          <div className="flex items-center gap-2 mb-2">
            <div className="p-2 bg-blue-600 rounded-lg text-white">
              <Bike size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Toronto Bikeshare AI</h1>
              <p className="text-xs text-gray-500">Demand Forecasting System</p>
            </div>
          </div>
          <div className="text-sm text-gray-600">
            {loading ? (
              <p>Loading model & stations...</p>
            ) : (
              <p>Monitoring <span className="font-semibold text-blue-600">{stations.length}</span> stations</p>
            )}
          </div>
        </div>

        {/* Map */}
        <Map
          stations={stations}
          onStationClick={setSelectedStation}
        />
      </div>
    </div>
  )
}

export default App
