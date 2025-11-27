import { useEffect, useState } from 'react'
import axios from 'axios'
import { X, TrendingUp, MapPin, Activity } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export function StationPanel({ station, onClose }) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        if (!station) return

        setLoading(true)
        axios.get(`http://127.0.0.1:8000/predict/${station.station_id}`)
            .then(res => {
                setData(res.data)
                setLoading(false)
            })
            .catch(err => {
                console.error(err)
                setLoading(false)
            })
    }, [station])

    if (!station) return null

    return (
        <div className="h-full flex flex-col bg-white">
            {/* Header */}
            <div className="p-6 border-b border-gray-100 flex justify-between items-start">
                <div>
                    <h2 className="text-xl font-bold text-gray-900 leading-tight">{station.name}</h2>
                    <div className="flex items-center gap-1 text-gray-500 text-sm mt-1">
                        <MapPin size={14} />
                        <span>ID: {station.station_id}</span>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-500"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-8">
                {/* Key Metrics */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                        <div className="flex items-center gap-2 text-blue-600 mb-1">
                            <TrendingUp size={16} />
                            <span className="text-xs font-semibold uppercase tracking-wider">Forecast</span>
                        </div>
                        <div className="text-2xl font-bold text-gray-900">
                            {station.predicted_demand.toFixed(1)}
                        </div>
                        <div className="text-xs text-gray-500">Trips expected tomorrow</div>
                    </div>

                    <div className="bg-gray-50 p-4 rounded-xl border border-gray-100">
                        <div className="flex items-center gap-2 text-gray-600 mb-1">
                            <Activity size={16} />
                            <span className="text-xs font-semibold uppercase tracking-wider">Capacity</span>
                        </div>
                        <div className="text-2xl font-bold text-gray-900">
                            {station.capacity || 'N/A'}
                        </div>
                        <div className="text-xs text-gray-500">Total docks</div>
                    </div>
                </div>

                {/* Chart */}
                <div>
                    <h3 className="text-sm font-semibold text-gray-900 mb-4">Recent Activity & Forecast</h3>
                    <div className="h-64 w-full">
                        {loading ? (
                            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
                                Loading history...
                            </div>
                        ) : data && data.history ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={[...data.history, { date: 'Tomorrow', count: data.forecast }]}>
                                    <defs>
                                        <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#2563eb" stopOpacity={0.8} />
                                            <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                                    <XAxis
                                        dataKey="date"
                                        tick={{ fontSize: 10 }}
                                        tickLine={false}
                                        axisLine={false}
                                        interval="preserveStartEnd"
                                    />
                                    <YAxis
                                        tick={{ fontSize: 10 }}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <Tooltip
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="count"
                                        stroke="#2563eb"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorCount)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-gray-400 text-sm">
                                No history data available
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
