"""Page 3: explain and visualize the existing workload-placement optimization."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scheduler import optimize, baseline


def render(sites, profiles, jobs):
    st.title('Optimization Control Center')
    st.caption('BIG BYTE / Page 3 / Dispatch, constraints and energy economics')
    st.info('24-hour, hourly, non-preemptive job placement. Existing facility demand and battery dispatch are fixed; jobs may reuse curtailed renewables or consume spare grid import capacity. All inputs are synthetic.')
    with st.sidebar:
        st.markdown('### Optimization controls')
        chosen = st.multiselect('Eligible sites', sites.site_name.tolist(), default=sites.site_name.tolist(), key='page3_sites')
        count = st.slider('Jobs to optimize', 1, len(jobs), len(jobs), key='page3_count')
        st.caption('Run the optimizer here or reuse a compatible Page 2 result.')
    eligible = sites[sites.site_name.isin(chosen)]
    selected_jobs = jobs.head(count)
    if eligible.empty:
        st.warning('Select at least one site.'); return
    ids = tuple(eligible.site_id)
    signature = (ids, count)
    cached = st.session_state.get('bigbyte_page3_result')
    previous = st.session_state.get('bigbyte_schedule')
    if previous and previous[4] == ids and previous[5] == count and not cached:
        cached = {'signature': signature, 'solution': previous[0], 'error': previous[1], 'base': previous[2], 'base_error': previous[3]}
    run = st.button('⚡ Run optimization', type='primary', use_container_width=True)
    if run:
        with st.spinner('Solving 24-hour mixed-integer placement model...'):
            solution, error = optimize(sites, profiles, selected_jobs, list(ids))
            base, base_error = baseline(sites, profiles, selected_jobs, list(ids))
        cached = {'signature': signature, 'solution': solution, 'error': error, 'base': base, 'base_error': base_error}
        st.session_state['bigbyte_page3_result'] = cached
        st.session_state['bigbyte_schedule'] = (solution, error, base, base_error, ids, count)
    if not cached or cached['signature'] != signature:
        st.warning('Run optimization to generate results for the selected sites and jobs.'); return
    if cached['error']:
        st.error(cached['error']); return
    sol, base = cached['solution'], cached['base']
    assigned, hourly = sol['assignments'].copy(), sol['hourly'].copy()
    cost = float(sol['cost_usd'])
    base_cost = float(base['cost_usd']) if base else None
    savings = (base_cost - cost) if base_cost is not None else None
    pct = 100*savings/base_cost if base_cost and base_cost > 0 else None
    total_job_mwh = float((selected_jobs.it_power_mw*selected_jobs.duration_hours).sum())
    renewable = float(hourly.renewable_reused_mw.sum())
    grid_mwh = float(hourly.incremental_grid_mw.sum())
    k1,k2,k3,k4 = st.columns(4)
    k1.metric('Jobs placed', f'{len(assigned)} / {len(selected_jobs)}')
    k2.metric('Incremental grid cost', f'${cost:,.2f}')
    k3.metric('Grid energy for jobs', f'{grid_mwh:,.2f} MWh')
    k4.metric('Curtailment reused', f'{renewable:,.2f} MWh')
    if base_cost is not None:
        st.success(f'Compared with first-fit scheduling: ${savings:,.2f} ({pct:,.1f}%) lower incremental grid electricity cost.' if savings >= 0 else f'Optimized cost exceeds the baseline by ${-savings:,.2f}. Review inputs.')
    else:
        st.warning('First-fit baseline could not schedule all jobs; no fair baseline cost comparison is available.')

    st.subheader('1. Where and when jobs run')
    timeline = assigned.copy()
    epoch = pd.Timestamp('2026-10-26')
    timeline['start'] = epoch + pd.to_timedelta(timeline.start_hour,unit='h')
    timeline['end'] = epoch + pd.to_timedelta(timeline.end_hour,unit='h')
    fig = px.timeline(timeline, x_start='start', x_end='end', y='job_id', color='site_name',
                      hover_data=['workload_type','it_power_mw','gpu_required','start_hour','end_hour'],
                      title='Non-preemptive job placement (synthetic UTC day)')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(height=max(440, 27*len(timeline)+130), legend_title_text='Site')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('2. Hourly incremental facility demand')
    plot = hourly.copy()
    plot['site_name'] = plot.site_id.map(sites.set_index('site_id').site_name)
    fig = px.area(plot, x='hour', y='incremental_facility_mw', color='site_name',
                  labels={'hour':'Hour (UTC)','incremental_facility_mw':'Additional facility power (MW)'},
                  title='Additional demand from scheduled jobs (IT power × PUE)')
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f'Job IT energy: {total_job_mwh:,.2f} MWh. Facility energy is higher because PUE accounts for cooling and other overhead.')

    st.subheader('3. How additional demand is supplied')
    supply = plot.groupby('hour',as_index=False)[['incremental_grid_mw','renewable_reused_mw']].sum()
    fig = go.Figure()
    fig.add_bar(x=supply.hour,y=supply.renewable_reused_mw,name='Previously curtailed renewables reused')
    fig.add_bar(x=supply.hour,y=supply.incremental_grid_mw,name='Additional grid imports')
    fig.update_layout(barmode='stack',xaxis_title='Hour (UTC)',yaxis_title='Incremental power (MW)',legend=dict(orientation='h',y=1.15))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Renewable reuse is limited by the existing synthetic curtailment at each site and hour. Battery charging/discharging is not changed by this model.')

    st.subheader('4. Constraints and capacity checks')
    base_profiles = profiles[profiles.site_id.isin(ids)][['site_id','hour','it_load_mw','grid_import_mw','curtailment_mw']].copy()
    check = plot.merge(base_profiles,on=['site_id','hour'],how='left').merge(
        eligible[['site_id','compute_capacity_mw','grid_import_limit_mw']],on='site_id',how='left')
    check['post_job_it_mw'] = check.it_load_mw + check.incremental_facility_mw / check.site_id.map(eligible.set_index('site_id').pue)
    check['post_job_grid_mw'] = check.grid_import_mw + check.incremental_grid_mw
    check['it_headroom_mw'] = check.compute_capacity_mw - check.post_job_it_mw
    check['grid_headroom_mw'] = check.grid_import_limit_mw - check.post_job_grid_mw
    violations = int(((check.it_headroom_mw < -1e-5) | (check.grid_headroom_mw < -1e-5)).sum())
    a,b,c = st.columns(3)
    a.metric('Capacity violations',violations)
    b.metric('Minimum spare IT power',f'{check.it_headroom_mw.min():.2f} MW')
    c.metric('Minimum spare grid import',f'{check.grid_headroom_mw.min():.2f} MW')
    site_filter=st.selectbox('Inspect site',eligible.site_name.tolist(),key='page3_inspect')
    sid=eligible.loc[eligible.site_name==site_filter,'site_id'].iloc[0]
    c=check[check.site_id==sid].sort_values('hour')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=c.hour,y=c.post_job_it_mw,name='IT load after jobs',mode='lines+markers'))
    fig.add_trace(go.Scatter(x=c.hour,y=c.compute_capacity_mw,name='IT capacity',line=dict(dash='dash')))
    fig.update_layout(title='IT capacity check',xaxis_title='Hour (UTC)',yaxis_title='IT power (MW)')
    st.plotly_chart(fig,use_container_width=True)
    with st.expander('Inspect all hourly constraints'):
        st.dataframe(check[['site_name','hour','post_job_it_mw','compute_capacity_mw','it_headroom_mw','post_job_grid_mw','grid_import_limit_mw','grid_headroom_mw']].round(4),hide_index=True,use_container_width=True)

    st.subheader('5. Cost and placement summary')
    if base_cost is not None:
        comparison = pd.DataFrame({'Method':['First-fit baseline','Big Byte optimized'],'Incremental cost (USD)':[base_cost,cost]})
        st.plotly_chart(px.bar(comparison,x='Method',y='Incremental cost (USD)',text_auto='.2f'),use_container_width=True)
    summary = assigned.groupby('site_name',as_index=False).agg(jobs=('job_id','count'),job_it_mwh=('it_power_mw',lambda x: 0.0))
    summary['job_it_mwh'] = summary.site_name.map(assigned.assign(job_mwh=assigned.it_power_mw*assigned.duration_hours).groupby('site_name').job_mwh.sum())
    st.dataframe(summary.round(3),hide_index=True,use_container_width=True)
    st.download_button('Download optimized assignments',assigned.to_csv(index=False),'bigbyte_page3_assignments.csv','text/csv',use_container_width=True)
    st.download_button('Download hourly energy dispatch',plot.to_csv(index=False),'bigbyte_page3_dispatch.csv','text/csv',use_container_width=True)
    st.caption('Scope: incremental grid-energy cost only; excludes GPU rental, network transfer, demand charges, battery degradation and emissions. Results depend entirely on synthetic assumptions.')
