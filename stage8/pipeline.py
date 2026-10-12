import argparse,subprocess,sys
from pathlib import Path
from common import write,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--gpu-benchmark',action='store_true');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parent;b=str(Path(a.baseline).resolve())
    steps=[('reference_metrics.py',['--baseline',b,'--output',str(out/'references')]),
        ('train_adapter.py',['--baseline',b,'--output',str(out/'adapter_training'),'--device',a.device]),
        ('evaluate_adapter.py',['--baseline',b,'--trained',str(out/'adapter_training'),'--output',str(out/'adapter_evaluation'),'--device',a.device]),
        ('metrics.py',['--baseline',b,'--extra-scores',str(out/'adapter_evaluation/adapter_human_scores.json'),'--output',str(out/'expanded_metrics.json')])]
    if a.gpu_benchmark:steps.append(('benchmark_gpu.py',['--baseline',b,'--trained',str(out/'adapter_training'),'--output',str(out/'fresh_gpu_benchmark.json')]))
    complete=[]
    for script,params in steps:
        print('START',script,flush=True)
        with (out/(script+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,str(root/script),*params],stdout=log,stderr=subprocess.STDOUT)
        assert r.returncode==0,f'{script} failed: see its saved log; no metric is silently omitted'
        complete.append(script);write(out/'pipeline_status.json',{'completed_steps':complete,'complete':len(complete)==len(steps),'protocol_sha256':digest(root/'protocol.json')})
        print('DONE',script,flush=True)
    if a.gpu_benchmark:
        print('START optional official PAC-S',flush=True)
        with (out/'pacscore_gpu.py.log').open('w') as log:
            r=subprocess.run([sys.executable,str(root/'pacscore_gpu.py'),'--baseline',b,'--output',str(out/'pacscore')],stdout=log,stderr=subprocess.STDOUT)
        write(out/'pacscore_status.json',{'completed':r.returncode==0,'exit_code':r.returncode,'log':'pacscore_gpu.py.log','required_core_pipeline_completed':True})
        print('PAC-S completed' if r.returncode==0 else 'PAC-S failed; saved explicit status and log',flush=True)
    write(out/'output_sha256.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='output_sha256.json'})

if __name__=='__main__':main()
