"""24h non-preemptive MILP placement, using existing telemetry as a fixed baseline.

Incremental job demand can consume previously curtailed renewables, then grid headroom.
BESS dispatch and existing loads are fixed; this is not a joint energy dispatch optimizer.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix


def prepare_options(sites, profiles, jobs, site_ids=None):
    sites=sites[sites.site_id.isin(site_ids or sites.site_id.tolist())].copy()
    profiles=profiles[profiles.site_id.isin(sites.site_id)].copy()
    rows=[]
    for _,j in jobs.iterrows():
        d=int(j.duration_hours)
        for _,s in sites.iterrows():
            p=profiles[profiles.site_id==s.site_id].set_index('hour')
            for start in range(int(j.arrival_hour),min(24-d,int(j.deadline_hour)-d)+1):
                if start<0: continue
                costs=[]; possible=True
                for h in range(start,start+d):
                    r=p.loc[h]
                    # Facility electrical load includes PUE.
                    additional=float(j.it_power_mw)*float(s.pue)
                    if float(r.it_load_mw)+float(j.it_power_mw)>float(s.compute_capacity_mw)+1e-7:
                        possible=False;break
                    renewable_used=min(additional,max(0,float(r.curtailment_mw)))
                    grid_used=additional-renewable_used
                    if float(r.grid_import_mw)+grid_used>float(s.grid_import_limit_mw)+1e-7:
                        possible=False;break
                    costs.append(grid_used*float(r.electricity_price_usd_mwh))
                if possible:
                    rows.append({'job_id':j.job_id,'site_id':s.site_id,'start_hour':start,
                                 'end_hour':start+d,'duration_hours':d,'it_power_mw':float(j.it_power_mw),
                                 'facility_power_mw':float(j.it_power_mw)*float(s.pue),
                                 'incremental_cost_usd':round(sum(costs),7)})
    return pd.DataFrame(rows)


def _feasible_constraints(options, sites, profiles, jobs):
    n=len(options)
    site_map=sites.set_index('site_id')
    profile_map={(r.site_id,int(r.hour)):r for r in profiles.itertuples()}
    # Conservative capacity: assume all additional load draws from grid unless curtailed renewable is available.
    # Shared curtailed renewable is constrained explicitly.
    keys=[('job',j) for j in jobs.job_id]
    for sid in sites.site_id:
        for h in range(24):
            keys.extend([('compute',sid,h),('grid',sid,h),('curtail',sid,h)])
    A=lil_matrix((len(keys),n),dtype=float)
    lb=np.full(len(keys),-np.inf); ub=np.full(len(keys),np.inf)
    key_idx={k:i for i,k in enumerate(keys)}
    for j in jobs.job_id:
        i=key_idx[('job',j)];lb[i]=ub[i]=1
    for sid in sites.site_id:
        s=site_map.loc[sid]
        for h in range(24):
            r=profile_map[(sid,h)]
            ub[key_idx[('compute',sid,h)]]=max(0,float(s.compute_capacity_mw)-float(r.it_load_mw))
            # Grid and renewable share constraint: additional facility load <= grid headroom + curtailed renewable.
            ub[key_idx[('grid',sid,h)]]=max(0,float(s.grid_import_limit_mw)-float(r.grid_import_mw))+max(0,float(r.curtailment_mw))
            ub[key_idx[('curtail',sid,h)]]=max(0,float(r.curtailment_mw))
    # For accurate marginal cost under shared curtailment, add per-site-hour grid allocation variables.
    # Each z is incremental grid power, with z >= total incremental facility load - curtailed renewables.
    # z <= grid headroom; cost = z * hourly price.
    sh=[(sid,h) for sid in sites.site_id for h in range(24)]
    m=len(sh)
    B=lil_matrix((len(keys)+m,n+m),dtype=float)
    B[:len(keys),:n]=A
    lower=np.concatenate([lb,np.full(m,-np.inf)])
    upper=np.concatenate([ub,np.zeros(m)])
    c=np.zeros(n+m)
    for k,o in enumerate(options.itertuples()):
        B[key_idx[('job',o.job_id)],k]=1
        for h in range(int(o.start_hour),int(o.end_hour)):
            B[key_idx[('compute',o.site_id,h)],k]=float(o.it_power_mw)
            B[key_idx[('grid',o.site_id,h)],k]=float(o.facility_power_mw)
            B[key_idx[('curtail',o.site_id,h)],k]=0  # retained as unused redundant constraint
    for t,(sid,h) in enumerate(sh):
        r=profile_map[(sid,h)]
        row=len(keys)+t
        # total incremental load - z <= curtailment
        for k,o in enumerate(options.itertuples()):
            if o.site_id==sid and int(o.start_hour)<=h<int(o.end_hour):
                B[row,k]=float(o.facility_power_mw)
        B[row,n+t]=-1
        upper[row]=max(0,float(r.curtailment_mw))
        c[n+t]=float(r.electricity_price_usd_mwh)
    lower_bounds=np.zeros(n+m)
    upper_bounds=np.concatenate([np.ones(n),np.array([max(0,float(profile_map[(sid,h)].grid_import_mw)*0 + float(site_map.loc[sid].grid_import_limit_mw)-float(profile_map[(sid,h)].grid_import_mw)) for sid,h in sh])])
    return c,Bounds(lower_bounds,upper_bounds),LinearConstraint(B.tocsr(),lower,upper),n,sh


def optimize(sites,profiles,jobs,site_ids=None):
    selected=sites[sites.site_id.isin(site_ids or sites.site_id.tolist())].copy()
    p=profiles[profiles.site_id.isin(selected.site_id)].copy()
    options=prepare_options(selected,p,jobs)
    if options.empty or set(options.job_id)!=set(jobs.job_id):
        return None,'At least one job has no feasible placement within its time window and selected sites.'
    c,bounds,constraints,n,sh=_feasible_constraints(options,selected,p,jobs)
    integrality=np.concatenate([np.ones(n,dtype=int),np.zeros(len(sh),dtype=int)])
    res=milp(c=c,integrality=integrality,bounds=bounds,constraints=constraints,
             options={'time_limit':30,'mip_rel_gap':0.001})
    if res.x is None:
        return None,f'No feasible solution found: {res.message}'
    assigned=options.iloc[np.flatnonzero(res.x[:n]>0.5)].copy()
    if len(assigned)!=len(jobs):
        return None,'Optimizer did not return a complete assignment.'
    assigned=assigned.drop(columns='incremental_cost_usd').merge(jobs[['job_id','workload_type','gpu_required','priority']],on='job_id')
    assigned['site_name']=assigned.site_id.map(selected.set_index('site_id').site_name)
    # Report realized marginal grid cost after renewable curtailment is shared across jobs.
    grid=pd.DataFrame(sh,columns=['site_id','hour'])
    grid['incremental_grid_mw']=np.maximum(0,res.x[n:])
    grid=grid.merge(p[['site_id','hour','electricity_price_usd_mwh','curtailment_mw']],on=['site_id','hour'])
    grid['hourly_cost_usd']=grid.incremental_grid_mw*grid.electricity_price_usd_mwh
    grid['incremental_facility_mw']=0.0
    for a in assigned.itertuples():
        mask=(grid.site_id==a.site_id)&(grid.hour>=a.start_hour)&(grid.hour<a.end_hour)
        grid.loc[mask,'incremental_facility_mw']+=a.facility_power_mw
    grid['renewable_reused_mw']=np.maximum(0,grid.incremental_facility_mw-grid.incremental_grid_mw)
    return {'assignments':assigned.sort_values(['start_hour','site_id','job_id']).reset_index(drop=True),
            'hourly':grid,'cost_usd':float(grid.hourly_cost_usd.sum()),'solver_message':res.message},None


def baseline(sites,profiles,jobs,site_ids=None):
    """First-fit earliest feasible schedule, reusing curtailed renewable before grid."""
    selected=sites[sites.site_id.isin(site_ids or sites.site_id.tolist())].copy()
    p=profiles[profiles.site_id.isin(selected.site_id)].copy()
    options=prepare_options(selected,p,jobs)
    used_it={(s,h):0.0 for s in selected.site_id for h in range(24)}
    used_facility={(s,h):0.0 for s in selected.site_id for h in range(24)}
    by_profile=p.set_index(['site_id','hour'])
    smap=selected.set_index('site_id')
    picks=[]
    for j in jobs.sort_values(['arrival_hour','deadline_hour','job_id']).itertuples():
        opts=options[options.job_id==j.job_id].sort_values(['start_hour','site_id'])
        choice=None
        for o in opts.itertuples():
            okay=True
            for h in range(o.start_hour,o.end_hour):
                r=by_profile.loc[(o.site_id,h)]
                if float(r.it_load_mw)+used_it[(o.site_id,h)]+o.it_power_mw>float(smap.loc[o.site_id].compute_capacity_mw)+1e-7:okay=False;break
                fac=used_facility[(o.site_id,h)]+o.facility_power_mw
                if fac>float(r.curtailment_mw)+float(smap.loc[o.site_id].grid_import_limit_mw)-float(r.grid_import_mw)+1e-7:okay=False;break
            if okay:choice=o;break
        if choice is None:return None,'First-fit baseline could not place every job.'
        picks.append(choice)
        for h in range(choice.start_hour,choice.end_hour):
            used_it[(choice.site_id,h)]+=choice.it_power_mw
            used_facility[(choice.site_id,h)]+=choice.facility_power_mw
    cost=0.0
    for (sid,h),fac in used_facility.items():
        r=by_profile.loc[(sid,h)]
        cost+=max(0,fac-float(r.curtailment_mw))*float(r.electricity_price_usd_mwh)
    return {'cost_usd':cost,'assignments':pd.DataFrame(picks)},None
