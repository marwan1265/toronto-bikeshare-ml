---
title: Toronto Bikeshare AI
emoji: 🚲
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Toronto Bikeshare AI Forecaster 🚲

An interactive dashboard that predicts daily demand for Toronto Bikeshare stations. Powered by a PyTorch GRU neural network, this app visualizes future trip counts across the city to help identify high-traffic areas and usage patterns.

## Features
*   **AI Forecasting**: Uses historical trip data to predict next-day demand.
*   **Interactive Map**: Color-coded heatmap (Green/Amber/Red) showing station activity.
*   **Station Insights**: Detailed breakdown of historical usage and capacity for every dock.

## Tech Stack

*   **Frontend**: React, Vite, Tailwind CSS, Leaflet (Maps), Recharts (Data Viz)
*   **Backend**: Python, FastAPI, Uvicorn
*   **Machine Learning**: PyTorch (GRU Model), Pandas, NumPy
*   **Deployment**: Docker, Hugging Face Spaces


## How it Works
The backend runs a **Gated Recurrent Unit (GRU)** model trained on historical Toronto Bikeshare ridership data. It serves predictions via a **FastAPI** server, which are consumed by a **React** frontend to render the interactive map.
