# Big Byte — AI Energy-Aware Scheduling MVP (v0.6 demo)

A five-page Streamlit prototype demonstrating multi-site AI workload scheduling, hourly electricity dispatch, and battery-aware optimization. **All data are synthetic.** No real-world savings, GPU compatibility, or production control are claimed.

## Deploy to Streamlit Community Cloud

1. Upload the **contents of this folder** to the root of a GitHub repository (`app.py`, `requirements.txt`, `data/`, etc.).
2. In Streamlit Community Cloud, create an app from that repository, branch `main`, entrypoint `app.py`.
3. Keep the app public for demo access. Open it in an incognito browser and test all five pages.

## Run locally

```bash
python -m venv .venv
# Activate the environment (Windows: .venv\Scripts\activate; macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Pages

1. Fleet Dashboard
2. Workload Scheduler
3. Optimization Control Center
4. Results & Savings Analysis
5. Joint BESS + AI Optimization (numeric 0–24 h job chart; mutually exclusive battery charge/discharge; optional comparable v0.6 baseline)

## Important limitations

The application uses synthetic 24-hour electricity, renewable, site and job data. The comparison baseline is a feasible conventional scheduling preference with optimized battery dispatch, **not** an observed customer baseline. Its savings are scenario-dependent. The solver uses a finite time limit and optimality tolerance. It does not model GPU hardware compatibility, network transfer costs, data locality, latency, demand charges, degradation, or real-time dispatch. Do not use it to control real electrical or computing equipment.

## Source and security

Do not upload `.env`, ngrok tokens, API keys, or private customer data. No credentials are required for the included synthetic demonstration.
