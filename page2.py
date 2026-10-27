import streamlit as st
import pandas as pd
import plotly.express as px
from scheduler import optimize,baseline


def render(sites,profiles,jobs):
    st.title('AI Workload Scheduler')
    st.caption('BIG BYTE / Page 2 / 24-hour non-preemptive placement optimization')
    st.info('Uses fixed synthetic baseline telemetry. Additional jobs consume curtailed renewable energy first, then available grid headroom. Existing BESS dispatch is not reoptimized.')
    with st.sidebar:
        st.markdown('### Scheduler controls')
        chosen=st.multiselect('Eligible sites',sites.site_name.tolist(),default=sites.site_name.tolist(),key='scheduler_sites')
        count=st.slider('Number of demo jobs',1,len(jobs),len(jobs))
    eligible=sites[sites.site_name.isin(chosen)]
    subset=jobs.head(count).copy()
    a,b,c,d=st.columns(4)
    a.metric('Available sites',len(eligible))
    b.metric('Jobs in queue',len(subset))
    c.metric('Total job energy',f'{subset.energy_mwh.sum():.2f} MWh IT')
    d.metric('GPU-hours',f'{(subset.gpu_required*subset.duration_hours).sum():,.0f}')
    st.subheader('Workload queue')
    st.dataframe(subset[['job_id','workload_type','gpu_required','it_power_mw','duration_hours','arrival_hour','deadline_hour','priority']],hide_index=True,use_container_width=True)
    st.caption('Arrival and deadline hours are within one synthetic UTC day. All jobs must finish by their deadline.')
    if not len(eligible):
        st.warning('Choose at least one site.');return
    if st.button('⚡ Optimize workload placement',type='primary',use_container_width=True):
        with st.spinner('Solving mixed-integer scheduling model...'):
            solution,error=optimize(sites,profiles,subset,eligible.site_id.tolist())
            base,base_error=baseline(sites,profiles,subset,eligible.site_id.tolist())
            st.session_state['bigbyte_schedule']=(solution,error,base,base_error,tuple(eligible.site_id),count)
    cached=st.session_state.get('bigbyte_schedule')
    if not cached:
        st.info('Click **Optimize workload placement** to generate a schedule.');return
    solution,error,base,base_error,ids,prev_count=cached
    if ids!=tuple(eligible.site_id) or prev_count!=count:
        st.warning('Inputs changed. Click Optimize again to update the results.');return
    if error:
        st.error(error);return
    assigned=solution['assignments'];hourly=solution['hourly']
    st.subheader('Optimization results')
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Jobs scheduled',f'{len(assigned)}/{len(subset)}')
    k2.metric('Incremental grid cost',f'${solution["cost_usd"]:,.2f}')
    k3.metric('Curtailed renewables reused',f'{hourly.renewable_reused_mw.sum():,.2f} MWh')
    if base:
        improvement=(base['cost_usd']-solution['cost_usd'])/base['cost_usd']*100 if base['cost_usd']>0 else 0
        k4.metric('Cost vs first-fit baseline',f'{improvement:,.1f}%')
    else:k4.metric('Baseline','Not feasible')
    st.caption('Cost covers only the *additional demo jobs*, not the pre-existing facility load. Renewable energy reused from previously curtailed production is assigned zero incremental energy price in this illustrative model.')
    st.subheader('Assigned workload timeline')
    plot=assigned.copy()
    plot['start']=pd.to_datetime('2026-10-26')+pd.to_timedelta(plot.start_hour,unit='h')
    plot['finish']=pd.to_datetime('2026-10-26')+pd.to_timedelta(plot.end_hour,unit='h')
    fig=px.timeline(plot,x_start='start',x_end='finish',y='job_id',color='site_name',hover_data=['workload_type','it_power_mw','gpu_required'],title='Optimized placement by job')
    fig.update_yaxes(autorange='reversed');fig.update_layout(height=max(430,26*len(plot)+140),xaxis_title='Synthetic time (UTC)',yaxis_title='Job')
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Incremental energy requirements')
    p=hourly.copy();p['site_name']=p.site_id.map(sites.set_index('site_id').site_name)
    plot2=p.melt(id_vars=['site_name','hour'],value_vars=['incremental_grid_mw','renewable_reused_mw'],var_name='Source',value_name='MW')
    fig2=px.bar(plot2,x='hour',y='MW',color='Source',facet_row='site_name',barmode='stack',title='Additional jobs: hourly grid versus reused curtailed renewables')
    fig2.update_layout(height=230*len(eligible)+120)
    st.plotly_chart(fig2,use_container_width=True)
    if base:
        st.subheader('Cost comparison')
        comparison=pd.DataFrame({'Method':['First-fit baseline','Big Byte optimized'],'Incremental energy cost (USD)':[base['cost_usd'],solution['cost_usd']]})
        st.bar_chart(comparison.set_index('Method'))
    st.subheader('Schedule table')
    cols=['job_id','workload_type','site_name','start_hour','end_hour','duration_hours','gpu_required','it_power_mw','priority']
    st.dataframe(assigned[cols],hide_index=True,use_container_width=True)
    st.download_button('Download optimized schedule (CSV)',assigned.to_csv(index=False),'bigbyte_optimized_schedule.csv','text/csv')
    st.caption('Model limitations: no network/data transfer constraints, no migration, no GPU inventory mapping, no battery redispatch, and no pricing for unused on-site renewable generation. All data are synthetic.')
