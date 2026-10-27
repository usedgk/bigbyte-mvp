from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Big Byte | Fleet Dashboard', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
ROOT=Path(__file__).resolve().parent
@st.cache_data
def load_data():
    return (pd.read_csv(ROOT/'data/sites.csv'),pd.read_csv(ROOT/'data/hourly_profiles.csv'),pd.read_csv(ROOT/'data/jobs.csv'))
sites,profiles,jobs=load_data()

with st.sidebar:
    page=st.radio('Navigation',['Fleet Dashboard','Workload Scheduler','Optimization Control Center','Results & Savings Analysis','Joint BESS + AI Optimization'],key='bigbyte_page')
if page=='Joint BESS + AI Optimization':
    from page5 import render
    render(sites,profiles,jobs)
    st.stop()
if page=='Results & Savings Analysis':
    from page4 import render
    render(sites,profiles,jobs)
    st.stop()
if page=='Optimization Control Center':
    from page3 import render
    render(sites,profiles,jobs)
    st.stop()
if page=='Workload Scheduler':
    from page2 import render
    render(sites,profiles,jobs)
    st.stop()

st.markdown('''<style>
.block-container {padding-top:1.4rem;max-width:1450px;}
[data-testid="stMetric"] {background:#151e31;border:1px solid #293650;border-radius:12px;padding:14px;}
[data-testid="stMetricLabel"] {color:#9aacc8;}
.sitebox {border:1px solid #2b3b54;border-radius:13px;padding:17px;background:#131d2f;margin-bottom:12px;}
.sitebox h3 {margin:0 0 7px 0;font-size:1.1rem;}
.muted {color:#9badc5;font-size:0.86rem;}
.status {color:#6ee7b7;font-weight:700;}
</style>''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## ⚡ BIG BYTE')
    st.caption('Compute where the power is.')
    st.divider()
    st.markdown('### Fleet controls')
    selected_names=st.multiselect('Sites',sites.site_name.tolist(),default=sites.site_name.tolist())
    hour=st.slider('Snapshot hour (UTC, synthetic)',0,23,12)
    st.divider()
    st.info('Demo mode: all energy and load data are synthetic. Job scheduling and optimization are not active.')
    st.download_button('Download hourly profiles',profiles.to_csv(index=False),'bigbyte_hourly_profiles.csv','text/csv',use_container_width=True)
    st.download_button('Download demo jobs',jobs.to_csv(index=False),'bigbyte_jobs.csv','text/csv',use_container_width=True)

active_sites=sites[sites.site_name.isin(selected_names)]
if active_sites.empty:
    st.warning('Select at least one site in the sidebar.')
    st.stop()
active=profiles[profiles.site_id.isin(active_sites.site_id)]
snapshot=active[active.hour==hour]

st.title('Fleet Dashboard')
st.caption('BIG BYTE / Energy-aware distributed AI infrastructure / Synthetic 24-hour scenario')

m1,m2,m3,m4,m5=st.columns(5)
m1.metric('Sites online',len(active_sites))
m2.metric('IT compute capacity',f'{active_sites.compute_capacity_mw.sum():,.0f} MW')
m3.metric('Current IT demand',f'{snapshot.it_load_mw.sum():,.1f} MW')
m4.metric('Renewable generation',f'{snapshot.renewable_generation_mw.sum():,.1f} MW')
m5.metric('Grid import',f'{snapshot.grid_import_mw.sum():,.1f} MW')

st.subheader('Site overview')
cols=st.columns(len(active_sites))
for col,(_,s) in zip(cols,active_sites.iterrows()):
    r=snapshot[snapshot.site_id==s.site_id].iloc[0]
    with col:
        st.markdown(f'''<div class="sitebox"><h3>{s.site_name}</h3><div class="muted">{s.location} · {s.energy_source}</div><p><span class="status">● SIMULATED ONLINE</span></p><div class="muted">IT capacity: <b>{s.compute_capacity_mw:.0f} MW</b> · PUE: <b>{s.pue:.2f}</b></div><div class="muted">Renewable: <b>{r.renewable_generation_mw:.1f} MW</b> · Grid: <b>{r.grid_import_mw:.1f} MW</b></div><div class="muted">BESS: <b>{s.battery_capacity_mwh:.0f} MWh</b> · SOC: <b>{r.battery_soc_pct:.0f}%</b></div></div>''',unsafe_allow_html=True)
        st.progress(min(1.0,float(r.utilization_pct)/100),text=f'IT utilization {r.utilization_pct:.1f}%')
        st.caption(f'Energy price ${r.electricity_price_usd_mwh:.1f}/MWh')

st.subheader('Energy & compute telemetry')
tab1,tab2,tab3=st.tabs(['Generation and facility demand','Electricity prices','Battery state of charge'])
site_map=dict(zip(sites.site_id,sites.site_name))
chart=active.copy()
chart['Site']=chart.site_id.map(site_map)
with tab1:
    a,b=st.columns(2)
    with a:
        fig=px.line(chart,x='hour',y='renewable_generation_mw',color='Site',markers=True,title='Renewable generation by site',labels={'hour':'Hour (UTC)','renewable_generation_mw':'Power (MW)'})
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.line(chart,x='hour',y='facility_load_mw',color='Site',title='Facility electricity demand (IT × PUE)',labels={'hour':'Hour (UTC)','facility_load_mw':'Power (MW)'})
        st.plotly_chart(fig,use_container_width=True)
with tab2:
    fig=px.line(chart,x='hour',y='electricity_price_usd_mwh',color='Site',title='Synthetic hourly grid electricity price',labels={'hour':'Hour (UTC)','electricity_price_usd_mwh':'USD/MWh'})
    st.plotly_chart(fig,use_container_width=True)
with tab3:
    bess_ids=active_sites[active_sites.battery_capacity_mwh>0].site_id
    bess=chart[chart.site_id.isin(bess_ids)]
    if bess.empty: st.info('Selected sites have no BESS.')
    else:
        fig=px.line(bess,x='hour',y='battery_soc_pct',color='Site',title='Simulated battery SOC',labels={'hour':'Hour (UTC)','battery_soc_pct':'SOC (%)'})
        fig.update_yaxes(range=[0,100])
        st.plotly_chart(fig,use_container_width=True)

st.subheader('Fleet power balance')
fleet=active.groupby('hour',as_index=False)[['renewable_generation_mw','grid_import_mw','battery_discharge_mw','facility_load_mw','battery_charge_mw','curtailment_mw']].sum()
fig=go.Figure()
for name,key in [('Renewable generation','renewable_generation_mw'),('Grid import','grid_import_mw'),('Battery discharge','battery_discharge_mw')]:
    fig.add_trace(go.Bar(x=fleet.hour,y=fleet[key],name=name))
fig.add_trace(go.Scatter(x=fleet.hour,y=fleet.facility_load_mw+fleet.battery_charge_mw+fleet.curtailment_mw,name='Demand + charging + curtailment',mode='lines+markers',line=dict(width=3)))
fig.update_layout(barmode='stack',xaxis_title='Hour (UTC)',yaxis_title='Power (MW)',legend=dict(orientation='h',y=1.16),margin=dict(t=55))
st.plotly_chart(fig,use_container_width=True)
st.caption('Hourly balance: renewable generation + grid import + BESS discharge = facility demand + BESS charging + renewable curtailment (within CSV rounding).')

st.subheader('AI workload queue — preview')
a,b,c=st.columns(3)
a.metric('Demo jobs',len(jobs))
b.metric('Total requested GPU-hours',f'{(jobs.gpu_required*jobs.duration_hours).sum():,}')
c.metric('Scheduled jobs','0 (optimizer not connected)')
st.dataframe(jobs[['job_id','workload_type','gpu_required','it_power_mw','duration_hours','arrival_hour','deadline_hour','priority','status']],use_container_width=True,hide_index=True)
st.caption('The telemetry uses a synthetic aggregate utilization profile. Demo jobs are a separate unscheduled queue and are NOT yet included in the load calculation. The two will be linked in the optimization phase.')
