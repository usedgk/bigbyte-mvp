"""Page 4: auditable savings and results, based on the existing fixed-dispatch scheduler."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scheduler import optimize, baseline


def _hourly_from_assignments(assignments, sites, profiles):
    """Reconstruct incremental supply for a given schedule (including first-fit)."""
    p = profiles[profiles.site_id.isin(sites.site_id)].copy()
    p['incremental_facility_mw'] = 0.0
    for a in assignments.itertuples():
        fac = float(a.it_power_mw) * float(sites.set_index('site_id').loc[a.site_id, 'pue'])
        mask = (p.site_id == a.site_id) & (p.hour >= int(a.start_hour)) & (p.hour < int(a.end_hour))
        p.loc[mask, 'incremental_facility_mw'] += fac
    p['renewable_reused_mw'] = p[['incremental_facility_mw', 'curtailment_mw']].min(axis=1)
    p['incremental_grid_mw'] = (p.incremental_facility_mw - p.renewable_reused_mw).clip(lower=0)
    p['incremental_cost_usd'] = p.incremental_grid_mw * p.electricity_price_usd_mwh
    return p


def render(sites, profiles, jobs):
    st.title('Results & Savings Analysis')
    st.caption('BIG BYTE / Page 4 / First-fit baseline versus optimized placement')
    st.info('Synthetic 24-hour scenario. Only incremental electricity costs for newly scheduled jobs are compared. Existing facility loads and battery dispatch are fixed.')
    with st.sidebar:
        st.markdown('### Results controls')
        chosen = st.multiselect('Eligible sites', sites.site_name.tolist(), default=sites.site_name.tolist(), key='page4_sites')
        count = st.slider('Jobs to analyze', 1, len(jobs), len(jobs), key='page4_count')
        st.caption('Use the same selection as Pages 2–3 to reuse their solution.')
    eligible = sites[sites.site_name.isin(chosen)].copy()
    if eligible.empty:
        st.warning('Select at least one site.'); return
    ids = tuple(eligible.site_id.tolist())
    signature = (ids, count)
    cached = st.session_state.get('bigbyte_page4_result')
    prior = st.session_state.get('bigbyte_schedule')
    if prior and prior[4] == ids and prior[5] == count and (not cached or cached['signature'] != signature):
        cached = {'signature': signature, 'solution': prior[0], 'error': prior[1], 'base': prior[2], 'base_error': prior[3]}
    if st.button('⚡ Analyze results', type='primary', use_container_width=True):
        with st.spinner('Calculating optimized and baseline schedules...'):
            sol, error = optimize(sites, profiles, jobs.head(count), list(ids))
            base, base_error = baseline(sites, profiles, jobs.head(count), list(ids))
        cached = {'signature': signature, 'solution': sol, 'error': error, 'base': base, 'base_error': base_error}
        st.session_state['bigbyte_page4_result'] = cached
        st.session_state['bigbyte_schedule'] = (sol, error, base, base_error, ids, count)
    if not cached or cached['signature'] != signature:
        st.warning('Click **Analyze results** to calculate the savings report.'); return
    if cached['error']:
        st.error(cached['error']); return
    sol, base = cached['solution'], cached['base']
    opt = _hourly_from_assignments(sol['assignments'], eligible, profiles)
    # Reconstructed optimized cost avoids floating-point solver allocation artifacts.
    opt_cost = float(opt.incremental_cost_usd.sum())
    baseline_ok = base is not None and cached['base_error'] is None
    base_hourly = _hourly_from_assignments(base['assignments'], eligible, profiles) if baseline_ok else None
    base_cost = float(base_hourly.incremental_cost_usd.sum()) if baseline_ok else None
    savings = base_cost - opt_cost if baseline_ok else None
    pct = 100 * savings / base_cost if baseline_ok and base_cost > 0 else None
    total_it_mwh = float((jobs.head(count).it_power_mw * jobs.head(count).duration_hours).sum())
    facility_mwh = float(opt.incremental_facility_mw.sum())
    renewable_mwh = float(opt.renewable_reused_mw.sum())
    grid_mwh = float(opt.incremental_grid_mw.sum())

    st.subheader('Executive results')
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Jobs completed', f'{len(sol["assignments"])} / {count}')
    k2.metric('Optimized grid cost', f'${opt_cost:,.2f}')
    k3.metric('First-fit grid cost', f'${base_cost:,.2f}' if baseline_ok else 'Not feasible')
    k4.metric('Cost reduction', f'{pct:.1f}%' if pct is not None else 'N/A', delta=f'${savings:,.2f} saved' if savings is not None else None)
    if not baseline_ok:
        st.warning('The first-fit baseline could not place all jobs. A like-for-like savings percentage cannot be calculated.')
    elif base_cost == 0:
        st.info('Both scenarios have zero baseline grid-energy cost; percentage savings is undefined.')
    else:
        st.success(f'Optimized placement saves ${savings:,.2f} ({pct:.1f}%) in modeled incremental grid electricity cost relative to first-fit scheduling.')

    st.subheader('1. Electricity cost comparison')
    if baseline_ok:
        methods = pd.DataFrame({'Method':['First-fit baseline', 'Big Byte optimized'], 'Cost (USD)':[base_cost, opt_cost]})
        a, b = st.columns([2, 1])
        with a:
            fig = px.bar(methods, x='Method', y='Cost (USD)', color='Method', text_auto='.2f')
            fig.update_layout(showlegend=False, yaxis_range=[0, max(1.0,base_cost,opt_cost)*1.18])
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.metric('Daily scenario savings', f'${savings:,.2f}')
            st.metric('Illustrative 30-day repeat', f'${30*savings:,.2f}')
            st.caption('30-day figure is a straight extrapolation, not a forecast. Workload mix, prices and renewable output would change.')
    else:
        st.caption('Baseline comparison unavailable.')

    st.subheader('2. Energy sourcing and renewable utilization')
    a,b,c,d = st.columns(4)
    a.metric('Added IT energy', f'{total_it_mwh:.2f} MWh')
    b.metric('Added facility energy', f'{facility_mwh:.2f} MWh')
    c.metric('Renewables reused', f'{renewable_mwh:.2f} MWh')
    d.metric('Additional grid energy', f'{grid_mwh:.2f} MWh')
    mix = pd.DataFrame({'Source':['Reused curtailed renewables','Additional grid imports'], 'Energy (MWh)':[renewable_mwh, grid_mwh]})
    st.plotly_chart(px.pie(mix, names='Source', values='Energy (MWh)', hole=0.55, title='Energy supply for scheduled jobs'), use_container_width=True)
    st.caption(f'{100*renewable_mwh/facility_mwh:.1f}% of incremental facility electricity is supplied by previously curtailed renewables.' if facility_mwh else 'No incremental facility energy.')

    st.subheader('3. Hourly cost and grid-import comparison')
    hourly = opt.groupby('hour', as_index=False)[['incremental_cost_usd','incremental_grid_mw','renewable_reused_mw']].sum()
    hourly = hourly.rename(columns={'incremental_cost_usd':'Optimized cost (USD)','incremental_grid_mw':'Optimized grid import (MW)'})
    if baseline_ok:
        bh = base_hourly.groupby('hour', as_index=False)[['incremental_cost_usd','incremental_grid_mw']].sum()
        hourly = hourly.merge(bh, on='hour', how='left').rename(columns={'incremental_cost_usd':'First-fit cost (USD)','incremental_grid_mw':'First-fit grid import (MW)'})
    a,b=st.columns(2)
    with a:
        cols=['Optimized cost (USD)'] + (['First-fit cost (USD)'] if baseline_ok else [])
        fig=px.line(hourly.melt(id_vars='hour', value_vars=cols, var_name='Method', value_name='USD'), x='hour', y='USD', color='Method', markers=True, title='Incremental electricity cost per hour')
        st.plotly_chart(fig, use_container_width=True)
    with b:
        cols=['Optimized grid import (MW)'] + (['First-fit grid import (MW)'] if baseline_ok else [])
        fig=px.line(hourly.melt(id_vars='hour', value_vars=cols, var_name='Method', value_name='MW'), x='hour', y='MW', color='Method', markers=True, title='Incremental grid import per hour')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('4. Site-level allocation')
    site_map = eligible.set_index('site_id').site_name.to_dict()
    opt['site_name'] = opt.site_id.map(site_map)
    report = opt.groupby('site_name', as_index=False).agg(grid_mwh=('incremental_grid_mw','sum'), renewable_mwh=('renewable_reused_mw','sum'), grid_cost_usd=('incremental_cost_usd','sum'))
    job_count = sol['assignments'].groupby('site_name').size()
    report['jobs_assigned'] = report.site_name.map(job_count).fillna(0).astype(int)
    if baseline_ok:
        base_hourly['site_name'] = base_hourly.site_id.map(site_map)
        baseline_costs = base_hourly.groupby('site_name').incremental_cost_usd.sum()
        report['first_fit_cost_usd'] = report.site_name.map(baseline_costs).fillna(0)
        report['site_cost_difference_usd'] = report.first_fit_cost_usd - report.grid_cost_usd
    st.dataframe(report.round(3), hide_index=True, use_container_width=True)
    fig=px.bar(report, x='site_name', y=['grid_mwh','renewable_mwh'], barmode='stack', labels={'value':'Incremental energy (MWh)','site_name':'Site','variable':'Source'}, title='Incremental energy mix by site')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('5. Verification and exports')
    # Check hourly power headroom after all jobs are assigned.
    checks = opt.merge(eligible[['site_id','compute_capacity_mw','grid_import_limit_mw','pue']], on='site_id')
    checks['post_job_it_mw'] = checks.it_load_mw + checks.incremental_facility_mw / checks.pue
    checks['post_job_grid_mw'] = checks.grid_import_mw + checks.incremental_grid_mw
    violations = int(((checks.post_job_it_mw > checks.compute_capacity_mw+1e-5) | (checks.post_job_grid_mw > checks.grid_import_limit_mw+1e-5)).sum())
    late = sol['assignments'].merge(jobs[['job_id','deadline_hour','arrival_hour']], on='job_id')
    deadline_violations = int(((late.end_hour > late.deadline_hour) | (late.start_hour < late.arrival_hour)).sum())
    x,y,z=st.columns(3)
    x.metric('Power-capacity violations', violations)
    y.metric('Time-window violations', deadline_violations)
    z.metric('Jobs scheduled', len(late))
    if violations or deadline_violations:
        st.error('Constraint validation detected violations. Do not rely on these results until resolved.')
    else:
        st.success('All reported job assignments pass the hourly power-capacity and scheduling-window checks.')
    st.download_button('Download site results CSV', report.to_csv(index=False), 'bigbyte_page4_site_results.csv', 'text/csv', use_container_width=True)
    st.download_button('Download hourly comparison CSV', hourly.to_csv(index=False), 'bigbyte_page4_hourly_comparison.csv', 'text/csv', use_container_width=True)
    st.download_button('Download optimized job schedule CSV', sol['assignments'].to_csv(index=False), 'bigbyte_page4_job_schedule.csv', 'text/csv', use_container_width=True)
    st.caption('Limitations: synthetic inputs; no battery redispatch, network transfer cost, GPU inventory, demand charges, battery degradation, carbon intensity, or actual workload migration. Savings are incremental grid-energy costs only.')
