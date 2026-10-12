import argparse,csv
from pathlib import Path
from common import read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--expanded',required=True);p.add_argument('--references',required=True);p.add_argument('--pac');p.add_argument('--output',required=True);a=p.parse_args()
    tables=read(a.expanded)['cohorts'];refs=read(a.references)['cohorts']
    for cohort in tables:
        for name,metrics in refs[cohort].items():
            assert name not in tables[cohort];tables[cohort][name]=metrics
    if a.pac:
        pac=read(a.pac)
        for cohort in tables:tables[cohort]['PAC-S-ViTB32']=pac[cohort]
    write(a.output,{'scope':'Exploratory diagnostics on historical observed datasets. Reference-based methods receive five extra reference captions. gemma-original-literal768 duplicates the recomputed original768 baseline.','cohorts':tables})
    rows=[{'cohort':c,'method':n,**m} for c,t in tables.items() for n,m in t.items()]
    with Path(a.output).with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Combined',len(tables['unseen900']),'named score columns')

if __name__=='__main__':main()
