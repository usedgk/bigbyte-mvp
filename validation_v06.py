"""Comparable scenario and post-solve physical checks for the synthetic demo."""
from joint_optimizer import joint_optimize


def _solve(sites,profiles,jobs,site_ids,terminal_soc,**kwargs):
    sol,err=joint_optimize(sites,profiles,jobs,site_ids=site_ids,terminal_soc=terminal_soc,**kwargs)
    if err:raise RuntimeError(err)
    return sol


def compare_scenarios(sites,profiles,jobs,site_ids=None,terminal_soc=True):
    conventional=_solve(sites,profiles,jobs,site_ids,terminal_soc,placement_preference=True)
    baseline=_solve(sites,profiles,jobs,site_ids,terminal_soc,fixed_assignments=conventional['assignments'])
    joint=_solve(sites,profiles,jobs,site_ids,terminal_soc)
    return {'baseline':baseline,'joint':joint,
            'baseline_cost':baseline['total_grid_cost_usd'],
            'joint_cost':joint['total_grid_cost_usd']}


def validate_solution(sites,jobs,assignments,hourly,terminal_soc=True,tol=1e-4):
    errors=[]
    if len(assignments)!=len(jobs) or assignments.job_id.nunique()!=len(jobs):
        errors.append('Incomplete or duplicated job assignment')
    for j in jobs.itertuples():
        rows=assignments[assignments.job_id==j.job_id]
        if len(rows)!=1:
            errors.append(f'{j.job_id}: assignment count');continue
        a=rows.iloc[0]
        if a.start_hour<j.arrival_hour-tol or a.end_hour>j.deadline_hour+tol or abs((a.end_hour-a.start_hour)-j.duration_hours)>tol:
            errors.append(f'{j.job_id}: scheduling window')
    for s in sites.itertuples():
        rows=hourly[hourly.site_id==s.site_id].sort_values('hour')
        if len(rows)!=24:
            errors.append(f'{s.site_id}: missing hours');continue
        cap=float(s.battery_capacity_mwh); power=float(s.battery_power_mw)
        previous=cap*float(s.initial_soc_pct)/100
        for r in rows.itertuples():
            tag=f'{s.site_id}/h{r.hour}'
            if abs(r.balance_residual_mw)>tol:errors.append(tag+': power balance')
            if r.base_it_mw+r.added_it_mw>float(s.compute_capacity_mw)+tol:errors.append(tag+': IT capacity')
            if r.grid_mw>float(s.grid_import_limit_mw)+tol or r.grid_mw < -tol:errors.append(tag+': grid limit')
            if r.charge_mw>power+tol or r.discharge_mw>power+tol:errors.append(tag+': BESS power')
            if r.charge_mw>tol and r.discharge_mw>tol:errors.append(tag+': simultaneous BESS operations')
            if r.soc_end_mwh< (0.1*cap if cap else 0)-tol or r.soc_end_mwh>cap+tol:errors.append(tag+': SOC bounds')
            if abs(r.soc_start_mwh-previous)>tol:errors.append(tag+': SOC continuity')
            expected=previous+0.95*r.charge_mw-r.discharge_mw/0.95
            if abs(expected-r.soc_end_mwh)>tol:errors.append(tag+': SOC transition')
            previous=r.soc_end_mwh
        if terminal_soc and previous<cap*float(s.initial_soc_pct)/100-tol:errors.append(f'{s.site_id}: terminal SOC')
    return errors
