from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

sites = pd.DataFrame([
    dict(site_id='A', site_name='North Sea Offshore', location='North Sea', energy_source='Wind + BESS + Grid', compute_capacity_mw=80, renewable_capacity_mw=120, battery_capacity_mwh=200, battery_power_mw=50, initial_soc_pct=60, grid_import_limit_mw=60, pue=1.18, base_price_usd_mwh=95),
    dict(site_id='B', site_name='Frankfurt Grid Hub', location='Frankfurt, Germany', energy_source='Utility Grid', compute_capacity_mw=100, renewable_capacity_mw=0, battery_capacity_mwh=0, battery_power_mw=0, initial_soc_pct=0, grid_import_limit_mw=130, pue=1.25, base_price_usd_mwh=120),
    dict(site_id='C', site_name='Abu Dhabi Solar Hub', location='Abu Dhabi, UAE', energy_source='Solar + BESS + Grid', compute_capacity_mw=60, renewable_capacity_mw=90, battery_capacity_mwh=150, battery_power_mw=40, initial_soc_pct=70, grid_import_limit_mw=70, pue=1.22, base_price_usd_mwh=75),
])
sites.to_csv(DATA/'sites.csv', index=False)

# Fixed, reproducible 24-hour demo. Each hourly row is a one-hour energy interval.
hours = np.arange(24)
wind_cf = np.clip(0.48 + 0.18*np.sin(2*np.pi*(hours+4)/24) + rng.normal(0,0.055,24), 0.12,0.85)
solar_cf = np.maximum(0, np.sin(np.pi*(hours-6)/12)) * 0.82
price_shape = 12*np.sin(2*np.pi*(hours-15)/24) + 16*((hours>=17)&(hours<=21))
rows=[]
for s in sites.to_dict('records'):
    soc = s['initial_soc_pct']/100*s['battery_capacity_mwh']
    for h in hours:
        # Utilization is a synthetic fixed load profile; no job assignment yet.
        util = float(np.clip(0.59 + 0.10*np.sin(2*np.pi*(h-8)/24) + rng.normal(0,0.025), 0.40,0.83))
        it_mw = s['compute_capacity_mw']*util
        facility_mw = it_mw*s['pue']
        renewable_mw = (120*wind_cf[h] if s['site_id']=='A' else 90*solar_cf[h] if s['site_id']=='C' else 0.0)
        # Illustrative battery controller: absorb surplus renewable, discharge only to offset grid deficit.
        charge_mw=discharge_mw=0.0
        if s['battery_capacity_mwh']>0:
            surplus=max(0.0, renewable_mw-facility_mw)
            deficit=max(0.0, facility_mw-renewable_mw)
            charge_mw=min(surplus, s['battery_power_mw'], (s['battery_capacity_mwh']-soc)/0.95)
            soc += charge_mw*0.95
            discharge_mw=min(deficit, s['battery_power_mw'], soc*0.95)
            soc -= discharge_mw/0.95
        grid_import_mw=max(0.0, facility_mw+charge_mw-renewable_mw-discharge_mw)
        curtailment_mw=max(0.0, renewable_mw+discharge_mw+grid_import_mw-facility_mw-charge_mw)
        price=max(15.0, s['base_price_usd_mwh']+price_shape[h]+rng.normal(0,4))
        rows.append(dict(site_id=s['site_id'],hour=int(h),it_load_mw=round(it_mw,3),facility_load_mw=round(facility_mw,3),utilization_pct=round(util*100,2),renewable_generation_mw=round(renewable_mw,3),battery_charge_mw=round(charge_mw,3),battery_discharge_mw=round(discharge_mw,3),battery_soc_pct=round((100*soc/s['battery_capacity_mwh']) if s['battery_capacity_mwh'] else 0,2),grid_import_mw=round(grid_import_mw,3),curtailment_mw=round(curtailment_mw,3),electricity_price_usd_mwh=round(price,2),grid_limit_exceeded=bool(grid_import_mw>s['grid_import_limit_mw']+1e-6)))
profiles=pd.DataFrame(rows)
profiles.to_csv(DATA/'hourly_profiles.csv',index=False)

jobs=[]
for i in range(1,21):
    arrival=int(rng.integers(0,17))
    duration=int(rng.integers(2,7))
    slack=int(rng.integers(1,7))
    deadline=min(24, arrival+duration+slack)
    if deadline-arrival<duration: arrival=deadline-duration
    gpu=int(rng.choice([32,64,128,256,512]))
    # Synthetic 0.7 kW/GPU IT estimate; job energy does not include PUE.
    it_power_mw=round(gpu*0.0007,4)
    jobs.append(dict(job_id=f'JOB-{i:03d}',workload_type=str(rng.choice(['Training','Batch inference','Data processing','Simulation'])),gpu_required=gpu,it_power_mw=it_power_mw,duration_hours=duration,arrival_hour=arrival,deadline_hour=deadline,energy_mwh=round(it_power_mw*duration,4),priority=str(rng.choice(['Normal','High'],p=[0.8,0.2])),status='Unscheduled'))
pd.DataFrame(jobs).to_csv(DATA/'jobs.csv',index=False)
print('Generated:',*(f'{p.name} ({len(pd.read_csv(p))} rows)' for p in sorted(DATA.glob('*.csv'))),sep='\n - ')
