"""Page 5 — joint AI workload and BESS optimization."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joint_optimizer import joint_optimize, fixed_dispatch_comparison


def render(sites,profiles,jobs):
    st.title('Joint BESS + AI Optimization')
    st.caption('BIG BYTE / Page 5 / Co-optimize workload placement and battery dispatch')
    st.info('Synthetic 24-hour model. Re-optimizes battery dispatch from initial SOC; original hourly BESS telemetry is not imposed. Grid prices and renewables are synthetic.')
    with st.sidebar:
        st.markdown('### Joint optimization controls')
        chosen=st.multiselect('Sites for joint optimization',sites.site_name.tolist(),default=sites.site_name.tolist(),key='joint_sites')
        count=st.slider('Jobs to schedule',1,len(jobs),len(jobs),key='joint_count')
        terminal=st.checkbox('End batteries at or above initial SOC',value=True,key='joint_terminal')
        st.caption('Battery charge/discharge efficiencies: 95% each; minimum SOC: 10%; one-hour intervals.')
    active=sites[sites.site_name.isin(chosen)].copy()
    if active.empty:st.warning('Select at least one site.');return
    ids=tuple(active.site_id.tolist());signature=(ids,count,terminal)
    if st.button('⚡ Optimize workloads + BESS',type='primary',use_container_width=True):
        with st.spinner('Solving integrated mixed-integer energy and compute dispatch...'):
            sol,err=joint_optimize(sites,profiles,jobs.head(count),list(ids),terminal_soc=terminal)
            reference,ref_err=fixed_dispatch_comparison(sites,profiles,jobs.head(count),list(ids))
        st.session_state['joint_result']={'signature':signature,'sol':sol,'err':err,'reference':reference,'ref_err':ref_err}
    result=st.session_state.get('joint_result')
    if not result or result['signature']!=signature:
        st.warning('Click **Optimize workloads + BESS** to run the integrated model.');return
    if result['err']:
        st.error(result['err']);return
    sol=result['sol'];h=sol['hourly'];a=sol['assignments']
    ref=result['reference'];ref_ok=ref is not None and result['ref_err'] is None
    delta=ref['total_grid_cost_usd']-sol['total_grid_cost_usd'] if ref_ok else None
    st.subheader('Integrated optimization results')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Jobs placed',f'{len(a)}/{count}')
    c2.metric('Joint total grid cost',f'${sol["total_grid_cost_usd"]:,.2f}')
    c3.metric('Fixed-dispatch total cost',f'${ref["total_grid_cost_usd"]:,.2f}' if ref_ok else 'N/A')
    c4.metric('Difference vs fixed dispatch',f'${delta:,.2f}' if delta is not None else 'N/A',
              delta=f'{100*delta/ref["total_grid_cost_usd"]:.2f}% lower' if ref_ok and ref['total_grid_cost_usd'] else None)
    st.caption('Both costs include existing facility grid demand and the newly scheduled jobs. The reference keeps original BESS dispatch fixed. A negative difference means joint optimization costs more.')
    if not ref_ok:st.warning('Fixed-dispatch comparison unavailable: '+str(result['ref_err']))
    st.subheader('1. Hourly energy balance')
    fleet=h.groupby('hour',as_index=False)[['renewable_mw','grid_mw','discharge_mw','charge_mw','curtail_mw','base_facility_mw','added_facility_mw']].sum()
    fig=go.Figure()
    for label,col in [('Renewable generation','renewable_mw'),('Grid import','grid_mw'),('BESS discharge','discharge_mw')]:
        fig.add_bar(x=fleet.hour,y=fleet[col],name=label)
    fig.add_scatter(x=fleet.hour,y=fleet.base_facility_mw+fleet.added_facility_mw+fleet.charge_mw+fleet.curtail_mw,
                    name='Facility + charging + curtailment',mode='lines+markers')
    fig.update_layout(barmode='stack',xaxis_title='Hour',yaxis_title='MW')
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('2. Battery charging, discharging and SOC')
    batteries=h[h.site_id.isin(active[active.battery_capacity_mwh>0].site_id)]
    if batteries.empty:st.info('No batteries in selected sites.')
    else:
        left,right=st.columns(2)
        with left:
            bat=batteries.melt(id_vars=['site_name','hour'],value_vars=['charge_mw','discharge_mw'],var_name='Dispatch',value_name='MW')
            st.plotly_chart(px.line(bat,x='hour',y='MW',color='site_name',line_dash='Dispatch',title='Battery charge / discharge power'),use_container_width=True)
        with right:
            fig=px.line(batteries,x='hour',y='soc_end_pct',color='site_name',markers=True,title='Battery end-of-hour SOC (%)')
            fig.update_yaxes(range=[0,100]);st.plotly_chart(fig,use_container_width=True)
    st.subheader('3. AI workload placement')
    fig=go.Figure()
    for site,subset in a.groupby('site_name',sort=False):
        fig.add_trace(go.Bar(
            x=(subset.end_hour-subset.start_hour).astype(float).tolist(),
            y=subset.job_id.astype(str).tolist(),
            base=subset.start_hour.astype(float).tolist(),
            orientation='h',name=str(site),
            customdata=subset[['start_hour','end_hour']].to_numpy(),
            hovertemplate='Job: %{y}<br>Start: %{customdata[0]} h<br>End: %{customdata[1]} h<extra>%{fullData.name}</extra>'))
    fig.update_layout(title='AI workload placement',xaxis=dict(title='Hour (UTC)',type='linear',range=[0,24],dtick=2),
                      yaxis=dict(title='Job',type='category',autorange='reversed'),
                      barmode='overlay',height=max(450,30*len(a)))
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(a[['job_id','site_name','start_hour','end_hour','it_power_mw','workload_type']],hide_index=True,use_container_width=True)
    st.subheader('4. Dispatch verification')
    max_balance=float(h.balance_residual_mw.abs().max())
    st.metric('Maximum hourly power-balance residual',f'{max_balance:.8f} MW')
    if max_balance<1e-5:st.success('Power balance verified for all site-hours.')
    else:st.error('Power-balance residual exceeds tolerance; investigate before using results.')
    summary=h.groupby('site_name',as_index=False).agg(grid_mwh=('grid_mw','sum'),charge_mwh=('charge_mw','sum'),discharge_mwh=('discharge_mw','sum'),curtailed_mwh=('curtail_mw','sum'),grid_cost_usd=('grid_cost_usd','sum'))
    st.dataframe(summary,hide_index=True,use_container_width=True)
    st.download_button('Download optimized job assignments',a.to_csv(index=False),'bigbyte_joint_jobs.csv','text/csv')
    st.download_button('Download hourly joint dispatch',h.to_csv(index=False),'bigbyte_joint_dispatch.csv','text/csv')
    st.caption('Model limitations: synthetic inputs; no battery degradation, grid export, demand charges, data-transfer cost, GPU type compatibility, or network latency. No commercial savings are established.')
    st.subheader('5. v0.6: Baseline versus joint optimization')
    st.caption('Conventional placement preference (site order and earliest feasible start), with separately optimized BESS dispatch, versus joint cost optimization. Same constraints and synthetic inputs.')
    if st.button('Run comparable baseline and joint scenarios',key='v06_compare'):
        from validation_v06 import compare_scenarios, validate_solution
        with st.spinner('Solving and validating both scenarios...'):
            try:
                comp=compare_scenarios(sites,profiles,jobs.head(count),list(ids),terminal)
                issues_baseline=validate_solution(active,jobs.head(count),comp['baseline']['assignments'],comp['baseline']['hourly'],terminal)
                issues_joint=validate_solution(active,jobs.head(count),comp['joint']['assignments'],comp['joint']['hourly'],terminal)
                st.session_state['v06_result']={'signature':signature,'comp':comp,'errors':issues_baseline+issues_joint}
            except Exception as exc:
                st.error(f'Comparison could not be completed: {exc}')
    comparison=st.session_state.get('v06_result')
    if comparison and comparison['signature']==signature:
        if comparison['errors']:
            st.error('Validation failed; do not use the savings numbers.')
            st.write(comparison['errors'])
        else:
            comp=comparison['comp']
            st.success('Both scenarios passed validation.')
            b,j=comp['baseline_cost'],comp['joint_cost']
            k1,k2,k3=st.columns(3)
            k1.metric('Conventional + optimized BESS',f'${b:,.2f}')
            k2.metric('Big Byte joint optimization',f'${j:,.2f}')
            k3.metric('Difference',f'{100*(b-j)/b:.3f}%' if b>0 else 'N/A')
            costs=pd.DataFrame({'Scenario':['Conventional','Big Byte'],'Electricity cost (USD)':[b,j]})
            st.plotly_chart(px.bar(costs,x='Scenario',y='Electricity cost (USD)',color='Scenario',title='Comparable scenario costs'),use_container_width=True)
            st.download_button('Download scenario costs',costs.to_csv(index=False),'bigbyte_v06_costs.csv','text/csv')
