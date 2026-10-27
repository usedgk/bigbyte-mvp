"""Joint non-preemptive AI placement + hourly battery dispatch (synthetic 24h MILP).

All MW variables are hourly average MW; SOC is MWh at hour boundaries.
Renewable generation and existing facility demand are fixed inputs; historic battery
telemetry is NOT carried over, to avoid double-counting existing dispatch.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from scheduler import prepare_options, optimize as fixed_optimize


def joint_optimize(sites, profiles, jobs, site_ids=None, terminal_soc=True, time_limit=45, fixed_assignments=None, placement_preference=False):
    selected = sites[sites.site_id.isin(site_ids if site_ids is not None else sites.site_id)].copy()
    p = profiles[profiles.site_id.isin(selected.site_id)].copy()
    opts = prepare_options_relaxed(selected, jobs)
    if opts.empty or set(opts.job_id) != set(jobs.job_id):
        return None, 'A job has no eligible site/time placement.'
    sites_map = selected.set_index('site_id')
    telemetry = p.set_index(['site_id', 'hour'])
    sh = [(sid,h) for sid in selected.site_id for h in range(24)]
    n = len(opts)
    # Decision vars: job binaries; grid, charge, discharge, curtail for each site-hour;
    # SOC for each site at each boundary 0..24.
    index = {}
    for k in range(n): index[('job',k)] = len(index)
    for sid,h in sh:
        for v in ('grid','charge','discharge','curtail','mode'):
            index[(v,sid,h)] = len(index)
    for sid in selected.site_id:
        for h in range(25): index[('soc',sid,h)] = len(index)
    nv=len(index)
    c=np.zeros(nv); lo=np.zeros(nv); hi=np.full(nv,np.inf)
    integrality=np.zeros(nv,dtype=int)
    for k in range(n):
        integrality[k]=1;hi[k]=1
    for sid,h in sh:
        integrality[index[('mode',sid,h)]]=1
        hi[index[('mode',sid,h)]]=1
    for sid,h in sh:
        s=sites_map.loc[sid]; r=telemetry.loc[(sid,h)]
        hi[index[('grid',sid,h)]]=float(s.grid_import_limit_mw)
        hi[index[('charge',sid,h)]]=float(s.battery_power_mw)
        hi[index[('discharge',sid,h)]]=float(s.battery_power_mw)
        hi[index[('curtail',sid,h)]]=float(r.renewable_generation_mw)
        c[index[('grid',sid,h)]]=float(r.electricity_price_usd_mwh)
        # Tiny cycling penalty discourages simultaneous charging/discharging degeneracy.
        c[index[('charge',sid,h)]]=0.001
        c[index[('discharge',sid,h)]]=0.001
    for sid in selected.site_id:
        s=sites_map.loc[sid];cap=float(s.battery_capacity_mwh)
        for h in range(25):
            lo[index[('soc',sid,h)]]=0.1*cap if cap else 0
            hi[index[('soc',sid,h)]]=cap
        initial=cap*float(s.initial_soc_pct)/100
        lo[index[('soc',sid,0)]]=hi[index[('soc',sid,0)]]=initial
        if terminal_soc:
            lo[index[('soc',sid,24)]]=max(lo[index[('soc',sid,24)]],initial)
    rows=[];lbs=[];ubs=[]
    def constraint(coeff,lower=-np.inf,upper=np.inf):
        rows.append(coeff);lbs.append(lower);ubs.append(upper)
    for jid in jobs.job_id:
        constraint({index[('job',k)]:1 for k,o in enumerate(opts.itertuples()) if o.job_id==jid},1,1)
    active={(sid,h):[] for sid,h in sh}
    for k,o in enumerate(opts.itertuples()):
        for h in range(int(o.start_hour),int(o.end_hour)):
            active[(o.site_id,h)].append((k,float(o.it_power_mw),float(o.facility_power_mw)))
    for sid,h in sh:
        s=sites_map.loc[sid];r=telemetry.loc[(sid,h)]
        # Compute headroom: base IT + new IT <= rated IT capacity.
        constraint({index[('job',k)]:it for k,it,_ in active[(sid,h)]},
                   upper=float(s.compute_capacity_mw)-float(r.it_load_mw))
        # Grid + renewable + discharge = existing facility + added jobs + charge + curtail.
        coeff={index[('grid',sid,h)]:1,index[('discharge',sid,h)]:1,
               index[('charge',sid,h)]:-1,index[('curtail',sid,h)]:-1}
        for k,_,fac in active[(sid,h)]: coeff[index[('job',k)]]=-fac
        constraint(coeff,lower=float(r.facility_load_mw)-float(r.renewable_generation_mw),
                   upper=float(r.facility_load_mw)-float(r.renewable_generation_mw))
        # SOC_{t+1} = SOC_t + eta_charge*charge - discharge/eta_discharge.
        if float(s.battery_capacity_mwh)>0:
            constraint({index[('soc',sid,h+1)]:1,index[('soc',sid,h)]:-1,
                        index[('charge',sid,h)]:-0.95,index[('discharge',sid,h)]:1/0.95},0,0)
        else:
            constraint({index[('soc',sid,h+1)]:1,index[('soc',sid,h)]:-1},0,0)
    # Battery mode: charging and discharging are mutually exclusive.
    for sid,h in sh:
        power=float(sites_map.loc[sid].battery_power_mw)
        constraint({index[('charge',sid,h)]:1, index[('mode',sid,h)]:-power},upper=0)
        constraint({index[('discharge',sid,h)]:1, index[('mode',sid,h)]:power},upper=power)
    if fixed_assignments is not None:
        fixed={str(r.job_id):(str(r.site_id),int(r.start_hour)) for r in fixed_assignments.itertuples()}
        if set(fixed)!=set(map(str,jobs.job_id)):
            return None, 'Fixed schedule must contain exactly one placement for each job.'
        for k,o in enumerate(opts.itertuples()):
            selected_option=(str(o.site_id),int(o.start_hour))==fixed[str(o.job_id)]
            lo[index[('job',k)]]=hi[index[('job',k)]]=int(selected_option)
    if placement_preference:
        # Feasible conventional schedule: prefer earlier times and site order,
        # independent of electricity prices. BESS is still optimized afterward.
        rank={sid:i for i,sid in enumerate(selected.site_id)}
        for k,o in enumerate(opts.itertuples()):
            c[index[('job',k)]]=1000000*(rank[o.site_id]*25+int(o.start_hour))
        for sid,h in sh:
            c[index[('grid',sid,h)]]*=1e-8
    A=lil_matrix((len(rows),nv),dtype=float)
    for ri,co in enumerate(rows):
        for j,val in co.items(): A[ri,j]=val
    res=milp(c,integrality=integrality,bounds=Bounds(lo,hi),
             constraints=LinearConstraint(A.tocsr(),lbs,ubs),
             options={'time_limit':time_limit,'mip_rel_gap':0.002})
    if res.x is None:
        return None, 'Joint model infeasible or no solution within time limit: '+res.message
    if not res.success:
        return None, 'Solver did not finish successfully: '+res.message
    chosen=opts.iloc[[k for k in range(n) if res.x[k]>0.5]].copy()
    if len(chosen)!=len(jobs): return None,'Incomplete assignment returned.'
    chosen=chosen.merge(jobs[['job_id','workload_type','gpu_required','priority']],on='job_id')
    chosen['site_name']=chosen.site_id.map(selected.set_index('site_id').site_name)
    records=[]
    for sid,h in sh:
        r=telemetry.loc[(sid,h)];s=sites_map.loc[sid]
        added=sum(fac*res.x[index[('job',k)]] for k,_,fac in active[(sid,h)])
        it=sum(it*res.x[index[('job',k)]] for k,it,_ in active[(sid,h)])
        record={'site_id':sid,'hour':h,'site_name':s.site_name,'base_it_mw':float(r.it_load_mw),
                'added_it_mw':it,'base_facility_mw':float(r.facility_load_mw),
                'added_facility_mw':added,'renewable_mw':float(r.renewable_generation_mw),
                'price_usd_mwh':float(r.electricity_price_usd_mwh)}
        for v in ('grid','charge','discharge','curtail'):
            record[v+'_mw']=max(0,float(res.x[index[(v,sid,h)]]))
        record['soc_start_mwh']=float(res.x[index[('soc',sid,h)]])
        record['soc_end_mwh']=float(res.x[index[('soc',sid,h+1)]])
        record['soc_end_pct']=100*record['soc_end_mwh']/float(s.battery_capacity_mwh) if float(s.battery_capacity_mwh)>0 else 0
        record['grid_cost_usd']=record['grid_mw']*record['price_usd_mwh']
        record['balance_residual_mw']=(record['grid_mw']+record['renewable_mw']+record['discharge_mw']-
             record['base_facility_mw']-record['added_facility_mw']-record['charge_mw']-record['curtail_mw'])
        records.append(record)
    hourly=pd.DataFrame(records)
    return {'assignments':chosen.sort_values(['start_hour','site_id','job_id']).reset_index(drop=True),
            'hourly':hourly,'total_grid_cost_usd':float(hourly.grid_cost_usd.sum()),
            'solver_message':res.message,'objective_usd':float(res.fun), 'solver_gap':float(getattr(res,'mip_gap',float('nan')))},None


def prepare_options_relaxed(sites,jobs):
    """Time/site choices only; joint model determines power feasibility and dispatch."""
    rows=[]
    for j in jobs.itertuples():
        d=int(j.duration_hours)
        for s in sites.itertuples():
            if float(j.it_power_mw)>float(s.compute_capacity_mw):continue
            for start in range(max(0,int(j.arrival_hour)),min(24-d,int(j.deadline_hour)-d)+1):
                rows.append({'job_id':j.job_id,'site_id':s.site_id,'start_hour':start,
                    'end_hour':start+d,'duration_hours':d,'it_power_mw':float(j.it_power_mw),
                    'facility_power_mw':float(j.it_power_mw)*float(s.pue)})
    return pd.DataFrame(rows)


def fixed_dispatch_comparison(sites,profiles,jobs,site_ids=None):
    """Fixed original battery telemetry + optimized job placement, total grid cost."""
    solution,error=fixed_optimize(sites,profiles,jobs,site_ids)
    if error:return None,error
    p=profiles[profiles.site_id.isin(site_ids if site_ids is not None else sites.site_id)].copy()
    base_cost=float((p.grid_import_mw*p.electricity_price_usd_mwh).sum())
    return {'total_grid_cost_usd':base_cost+float(solution['cost_usd']),
            'incremental_cost_usd':float(solution['cost_usd']),'assignments':solution['assignments']},None
