"""Audit a completed native run and export its own Vietnamese research report.

No inference/training occurs here. Historical context is explicitly separate.
"""
import argparse
import base64
import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from bundle import CODE_ROOT, read, sha, verify_code, write
from common import seal, verify
from learning import partition_metrics
from replay import compare

MAIN = ['base', 'mean_single', 'uncertainty_single']
METRICS = [('sugar_accuracy_pct', 'Sugar accuracy (%)'),
           ('flickr_raw_tau_c', 'Flickr raw Kendall tau-c'),
           ('polaris_raw_tau_c', 'Polaris raw Kendall tau-c')]


def csvfile(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def flat(name, m):
    row = {'variant': name, 'sugar_accuracy_pct': 100*m['sugar']['accuracy'],
           'sugar_macro_pct': 100*m['sugar']['macro_accuracy'], 'sugar_ties': m['sugar']['ties']}
    for domain in ['flickr', 'polaris']:
        for key, value in m[domain].items():
            row[domain+'_'+('raw_tau_c' if key == 'raw_vote_tau_c' else key)] = value
    return row


def audit_timing(timing, core):
    verify(timing)
    if not read(timing/'status.json')['complete'] or not read(timing/'bootstrap_status.json')['complete']:
        raise ValueError('Timing producer incomplete')
    protocol = read(timing/'timing_protocol.json')
    if protocol['core_manifest_sha256'] != sha(core/'manifest_sha256.json'):
        raise ValueError('Timing belongs to a different core producer')
    rows = [r for r in read(timing/'timing_results.json') if r['backbone']=='gemma']
    samples = [r for r in read(timing/'timing_samples.json') if r['backbone']=='gemma']
    if len(rows)!=6 or len(samples)!=864 or len({(r['variant'],r['batch']) for r in rows})!=6:
        raise ValueError('Expected six Gemma timing rows / 864 calls')
    for row in rows:
        selected = sorted([s for s in samples if (s['variant'],s['batch'])==(row['variant'],row['batch'])],
                          key=lambda s:(s['round'],s['group_index']))
        batch = row['batch']; groups = 128//batch
        if {(s['round'],s['group_index']) for s in selected}!={(r,g) for r in range(2) for g in range(groups)}:
            raise ValueError('Incomplete or duplicated timing groups')
        values = np.array([s['ms_call'] for s in selected])
        if len(selected)!=2*groups or not np.isfinite(values).all() or (values<=0).any():
            raise ValueError('Invalid timing samples')
        baseline = sorted([s for s in samples if s['variant']=='base' and s['batch']==batch],
                          key=lambda s:(s['round'],s['group_index']))
        paired = values-np.array([s['ms_call'] for s in baseline])
        rng = np.random.default_rng(20261015)
        ids = rng.integers(groups,size=(2000,groups))
        medians = np.median(paired.reshape(2,groups)[:,ids].transpose(1,0,2).reshape(2000,-1),axis=1)
        expected = {'p50_ms_call':float(np.median(values)), 'p95_ms_call':float(np.percentile(values,95)),
                    'effective_throughput_pairs_s':float(batch*1000/np.median(values)),
                    'aggregate_throughput_pairs_s':float(batch*len(values)*1000/values.sum()),
                    'peak_allocated_mib':max(s['peak_allocated_mib'] for s in selected),
                    'paired_added_p50_ms_call':float(np.median(paired))}
        for key,value in expected.items():
            if not math.isclose(value,row[key],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Timing summary differs: '+key)
        if not np.allclose(np.percentile(medians,[2.5,97.5]),row['paired_added_ci95_ms_call'],rtol=1e-10,atol=1e-10):
            raise ValueError('Paired timing CI differs')
    return rows, {'complete':True,'rows':len(rows),'calls':len(samples), 'raw_samples_recomputed':True,
                  'paired_overhead_ci_recomputed':True,'protocol':protocol}


def build_report(core, output, timing=None, evidence_label='new_native_gpu_run'):
    """Require sealed producer inputs. Refuse to replace an existing report."""
    core, output = Path(core).resolve(), Path(output).resolve()
    timing = Path(timing).resolve() if timing else None
    verify_code(); verify(core)
    if output.exists():
        raise FileExistsError('Choose a new report directory')
    if not read(core/'status.json')['complete'] or not read(core/'bootstrap_status.json')['complete']:
        raise ValueError('Core producer incomplete')
    folder = core/'gemma'
    records = read(core/'records.json'); test = [r for r in records if r['split']=='test']
    sugar = [r for r in test if r['domain']=='sugar']
    mapping = read(folder/'prediction_mapping.json')
    if mapping['records_sha256']!=sha(core/'records.json') or mapping['test_record_ids']!=[r['id'] for r in test] or mapping['sugar_negative_ids']!=[r['id'] for r in sugar]:
        raise ValueError('Prediction IDs do not match records')
    metrics = read(folder/'metrics.json'); selection = read(folder/'selection.json')
    protocol = read(core/'protocol.json'); receipt = read(folder/'model_receipt.json')
    if any(selection[p]['test_used_for_selection'] for p in ['mean','uncertainty']):
        raise ValueError('Report requires validation-only selection')
    if receipt['dtype']!='torch.float32' or receipt['weights']!=read(CODE_ROOT/'PINNED.json')['gemma_weights']:
        raise ValueError('Report requires pinned Gemma FP32 weights')
    expected_names=set(MAIN)|{'ablation_'+s for s in ['sugar_only','human_only','no_preservation','full_control']}
    for phase in ['mean','uncertainty']:
        if selection[phase]['selection']['fallback_base']:
            compare(metrics[phase+'_single'],metrics['base'],'fallback_'+phase)
        else:
            expected_names|={phase+f'_seed{s}' for s in protocol['learning']['seeds']}
            expected_names.add(phase+'_ensemble3_diagnostic')
    if set(metrics)!=expected_names:raise ValueError('Missing/unexpected variants for selected or fallback phases')
    predictions = {}; recomputed = {}
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        for name,m in metrics.items():
            a,n = z[name+'_candidate'],z[name+'_negative']
            if a.shape!=(len(test),) or n.shape!=(len(sugar),) or not np.isfinite(a).all() or not np.isfinite(n).all():
                raise ValueError('Invalid prediction arrays: '+name)
            recomputed[name] = partition_metrics(test,a,n); compare(recomputed[name],m,name)
            predictions[name] = (a.copy(),n.copy())
    intervals = read(folder/'paired_bootstrap.json'); ci = []
    if set(intervals)!={'mean_single_vs_base','uncertainty_single_vs_base','uncertainty_vs_mean'}:
        raise ValueError('Required paired comparisons missing')
    for contrast,entry in intervals.items():
        left,right = ('mean_single','uncertainty_single') if contrast=='uncertainty_vs_mean' else ('base',contrast.removesuffix('_vs_base'))
        for name,item in entry['metrics'].items():
            domain,key = name.split('_',1)
            expected = metrics[right][domain][key]-metrics[left][domain][key]
            if not math.isclose(item['delta'],expected,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('CI point estimate differs: '+contrast+'/'+name)
            if not all(math.isfinite(v) for v in item['ci95']) or item['ci95'][0]>item['ci95'][1]:
                raise ValueError('Invalid CI bounds')
            ci.append({'contrast':contrast,'metric':name,'delta':item['delta'],'lower95':item['ci95'][0],
                       'upper95':item['ci95'][1],'unit':item['unit'],'replicates':entry['replicates'],
                       'sampling_unit':entry['sampling_unit']})
    timings, timing_audit = audit_timing(timing,core) if timing else ([],{'complete':False,'reason':'128-image timing not supplied'})
    for row in timings:
        if row['dtype']!=receipt['dtype'] or row['gpu']!=receipt['gpu']:
            raise ValueError('Timing hardware/precision differs from core')
    counts=[]
    for domain in ['sugar','flickr','polaris']:
        for split in ['train','validation','test']:
            rs=[r for r in records if r['domain']==domain and r['split']==split]
            counts.append({'domain':domain,'split':split,'records':len(rs),'images':len({r['image'] for r in rs}),
                           'votes':sum(len(r.get('ratings',[])) for r in rs)})
    image_sets={s:{r['image'] for r in records if r['split']==s} for s in ['train','validation','test']}
    overlaps={a+'_'+b:len(image_sets[a]&image_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(overlaps.values()):raise ValueError('Pixel-image split overlap')
    main=[flat(n,metrics[n]) for n in MAIN]; all_rows=[flat(n,m) for n,m in metrics.items()]
    parameters=read(folder/'parameter_counts.json')
    for row in main:
        n=row['variant'];row['adapter_parameters']=parameters.get(n,0);row['adapter_fp32_kib']=4*parameters.get(n,0)/1024
        for batch in [1,8]:
            item=next((r for r in timings if r['variant']==n and r['batch']==batch),None)
            for key in ['p50_ms_call','p95_ms_call','effective_throughput_pairs_s','peak_allocated_mib','paired_added_p50_ms_call']:
                row[f'b{batch}_{key}']=item[key] if item else None
    # Paired errors use strict Sugar correctness, including ties as errors.
    ids=np.array([i for i,r in enumerate(test) if r['domain']=='sugar'])
    base_correct=predictions['base'][0][ids]>predictions['base'][1]
    error_rows=[];categories=[]
    for name in MAIN[1:]:
        correct=predictions[name][0][ids]>predictions[name][1]
        for category in sorted({r['category'] for r in sugar}):
            mask=np.array([r['category']==category for r in sugar]); a=base_correct[mask];b=correct[mask]
            categories.append({'variant':name,'category':category,'pairs':int(mask.sum()),
                               'base_accuracy_pct':float(100*a.mean()),'accuracy_pct':float(100*b.mean()),
                               'delta_pp':float(100*(b.mean()-a.mean())), 'fixed':int((~a&b).sum()),
                               'regressed':int((a&~b).sum()),'net_fixed':int(b.sum()-a.sum())})
        for j,row in enumerate(sugar):
            if base_correct[j]!=correct[j]:
                error_rows.append({'variant':name,'id':row['id'],'image':row['image'],'category':row['category'],
                                   'change':'fixed' if correct[j] else 'regressed',
                                   'caption':row['caption'],'negative_caption':row['negative_caption'],
                                   'base_margin':float(predictions['base'][0][ids[j]]-predictions['base'][1][j]),
                                   'candidate_margin':float(predictions[name][0][ids[j]]-predictions[name][1][j])})
    trials=[];histories=[]
    for phase,entry in selection.items():
        for index,trial in enumerate(entry['trials']):
            trials.append({'phase':phase,'config_index':index,**trial['config'],
                           'gain_sugar':trial['mean_normalized_gain'][0],'gain_flickr':trial['mean_normalized_gain'][1],
                           'gain_polaris':trial['mean_normalized_gain'][2],
                           'eligible':all(v>=0 for v in trial['mean_normalized_gain']),
                           'best_epochs':json.dumps(trial['best_epochs'])})
    for path in sorted((folder/'training').glob('*.json')):
        for entry in read(path).get('history',[]):
            histories.append({'run':path.stem,'epoch':entry['epoch'],'loss':entry['loss'],
                              'validation_objective':entry.get('objective')})
    seed_rows=[]
    for phase in ['mean','uncertainty']:
        for seed in protocol['learning']['seeds']:
            name=phase+f'_seed{seed}'
            if name in metrics:seed_rows.append({'phase':phase,'seed':seed,**flat(name,metrics[name])})
    output.mkdir(parents=True);figdir=output/'figures';figdir.mkdir()
    tables={'all_variants':all_rows,'main_metrics':main,'paired_ci':ci,'data_splits':counts,
            'sugar_categories':categories,'sugar_changed_pairs':error_rows,'validation_trials':trials,
            'training_history':histories,'seed_metrics':seed_rows,'timing128':timings,
            'cached_head_timing':read(folder/'cached_head_latency.json')}
    for name,rows in tables.items():csvfile(output/(name+'.csv'),rows)
    for name,value in [('selection',selection),('recomputed_metrics',recomputed),('runtime',read(core/'runtime.json')),
                       ('model_receipt',receipt),('protocol',protocol)]:write(output/(name+'.json'),value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[]
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight')
        fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig);figures.append(name)
    colors=['#62748a','#007f91','#bd671d']
    with plt.rc_context({'font.size':9,'axes.grid':True,'grid.alpha':.2}):
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,(key,label) in zip(axes,METRICS):
            ys=[r[key] for r in main];ax.plot(range(3),ys,'o-',color=colors[1])
            ax.set_xticks(range(3),['Base','Mean primary','Uncertainty'],rotation=15);ax.set_title(label)
            ax.margins(x=.15,y=.4)
            for i,y in enumerate(ys):ax.annotate(f'{y:.5f}',(i,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        fig.suptitle('This run: Gemma FP32 quality (different, truncated axes)');save(fig,'01_quality')
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,(key,factor,label) in zip(axes,[('sugar_accuracy',100,'Sugar delta (pp)'),('flickr_raw_vote_tau_c',1,'Flickr raw delta'),('polaris_raw_vote_tau_c',1,'Polaris raw delta')]):
            for y,contrast in enumerate(intervals):
                r=next(r for r in ci if r['contrast']==contrast and r['metric']==key)
                ax.plot([factor*r['lower95'],factor*r['upper95']],[y,y],lw=2);ax.scatter(factor*r['delta'],y,s=25)
            ax.axvline(0,color='#555',lw=.8);ax.set_yticks(range(len(intervals)),list(intervals),fontsize=7);ax.set_title(label)
        fig.suptitle('Paired image-cluster 95% CI; exploratory, unadjusted');save(fig,'02_paired_ci')
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        ablations=[r for r in all_rows if r['variant'].startswith('ablation_')]+[all_rows[0]]
        for ax,(key,label) in zip(axes,METRICS):
            ax.barh([r['variant'].replace('ablation_','') for r in ablations],[r[key] for r in ablations],color=colors[0]);ax.set_title(label)
        fig.suptitle('Fixed-config seed11 ablations; 15 epochs; separate from selected primary');save(fig,'03_ablation')
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        selected=[r for r in categories if r['variant']=='mean_single']
        axes[0].barh([r['category'] for r in selected],[r['delta_pp'] for r in selected],color=[colors[1] if r['delta_pp']>=0 else colors[2] for r in selected])
        axes[0].axvline(0,color='#555');axes[0].set_title('Sugar category change: mean - base (pp)')
        for i,domain in enumerate(['flickr','polaris']):
            axes[1].bar(np.arange(3)+i*.35,[100*metrics[n][domain]['caption_ranking_accuracy'] for n in MAIN],width=.35,label=domain)
        axes[1].set_xticks(np.arange(3)+.175,['Base','Mean','Uncertainty']);axes[1].set_ylim(0,100);axes[1].set_title('Within-image ranking accuracy (%)');axes[1].legend()
        save(fig,'04_domain_errors')
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,(key,label) in zip(axes,METRICS):
            for phase in ['mean','uncertainty']:
                rs=[r for r in seed_rows if r['phase']==phase]
                if rs:ax.plot([r['seed'] for r in rs],[r[key] for r in rs],'o-',label=phase)
            ax.set_title(label);ax.set_xlabel('Refit seed')
        if seed_rows:axes[-1].legend()
        fig.suptitle('Three-seed diagnostics; selected config per phase; fallback has no trained seeds');save(fig,'05_seed_variation')
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for phase in ['mean','uncertainty']:
            rs=[r for r in histories if r['run']==phase+'_seed11_refit']
            if rs:axes[0].plot([r['epoch'] for r in rs],[r['loss'] for r in rs],label=phase)
        axes[0].set_title('Selected primary seed11 refit loss');axes[0].set_xlabel('Epoch')
        if axes[0].lines:axes[0].legend()
        gain_matrix=np.array([[r['gain_sugar'],r['gain_flickr'],r['gain_polaris']] for r in trials])
        limit=max(1e-6,float(np.abs(gain_matrix).max()))
        im=axes[1].imshow(gain_matrix,aspect='auto',cmap='RdYlGn',vmin=-limit,vmax=limit)
        axes[1].set_xticks(range(3),['Sugar','Flickr','Polaris']);axes[1].set_yticks(range(len(trials)),[r['phase']+' cfg'+str(r['config_index']) for r in trials],fontsize=7)
        axes[1].set_title('Mean normalized validation gains; no test selection');fig.colorbar(im,ax=axes[1]);save(fig,'06_training_selection')
        if timings:
            fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for ax,key,label in zip(axes,['b1_p50_ms_call','b8_effective_throughput_pairs_s','b8_peak_allocated_mib'],['Native B1 p50 (ms/call)','Native B8 effective pairs/s','B8 peak allocated MiB']):
                ax.bar(range(3),[r[key] for r in main],color=colors);ax.set_xticks(range(3),['Base','Mean','Uncertainty'],rotation=15);ax.set_title(label)
            fig.suptitle('128 unique images; 864 calls; preprocessing + backbone + head; excludes disk/download');save(fig,'07_native_cost')
    # Small historical reference tables travel inside the embedded notebook code.
    reference=CODE_ROOT/'report_context/seven_backbone_tradeoffs.csv';historical=[]
    control_summary=[];control_ci=[]
    if reference.exists():
        context=read(reference.parent/'context_manifest.json')
        for name,digest in context['files'].items():
            if sha(reference.parent/name)!=digest:raise ValueError('Historical context bytes changed: '+name)
        for name in ['multiseed_summary','multiseed_per_seed','crossed_global_ci','seven_backbone_quality_ci']:
            source=reference.parent/(name+'.csv')
            if source.exists():
                with source.open(encoding='utf-8-sig',newline='') as stream:values=list(csv.DictReader(stream))
                csvfile(output/('historical_'+name+'.csv'),values)
                if name=='multiseed_summary':control_summary=values
                if name=='crossed_global_ci':control_ci=values
        with reference.open(encoding='utf-8-sig',newline='') as stream:
            historical=[r for r in csv.DictReader(stream) if r['variant']=='mean_single']
        csvfile(output/'historical_seven_backbones.csv',historical)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,quality,label in zip(axes,['sugar_accuracy_pct_up','flickr_raw_tau_c_up','polaris_raw_tau_c_up'],['Sugar (%)','Flickr raw tau-c','Polaris raw tau-c']):
            for r in historical:
                x=float(r['representative_b1_p50_ms']);y=float(r[quality]);ax.scatter(x,y,color=colors[0])
                offset={'siglip224':(6,10),'siglip256':(6,-13),'pacs':(6,-12),'gemma':(-44,6)}.get(r['backbone'],(6,3))
                ax.annotate(r['backbone'],(x,y),fontsize=7,xytext=offset,textcoords='offset points')
            if timings:
                ax.scatter(main[1]['b1_p50_ms_call'],main[1][dict(zip(['sugar_accuracy_pct_up','flickr_raw_tau_c_up','polaris_raw_tau_c_up'],[m[0] for m in METRICS]))[quality]],marker='*',s=100,color=colors[2],label='This run')
                ax.legend(fontsize=7)
            ax.set_xscale('log');ax.set_xlabel('B1 p50 ms (log scale)');ax.set_title(label)
        fig.suptitle('Historical 2026-10-16 context; different sessions/precisions; no causal speed claim');save(fig,'08_historical_tradeoff')
    # A single set of sections drives Markdown and a standalone HTML report.
    sections=[]
    def section(title,text='',rows=None,columns=None,figure=None):sections.append((title,text,rows,columns,figure))
    ci_main=[r for r in ci if r['contrast']=='mean_single_vs_base' and r['metric'] in ['sugar_accuracy','flickr_raw_vote_tau_c','polaris_raw_vote_tau_c']]
    findings=[]
    for r in ci_main:
        factor=100 if r['unit']=='fraction' else 1
        interpretation='CI nằm trên 0' if r['lower95']>0 else ('CI nằm dưới 0' if r['upper95']<0 else 'CI chứa 0; chưa đủ bằng chứng về dấu của cải thiện')
        findings.append(f"{r['metric']}: mean − base = {factor*r['delta']:+.6f}, CI95 [{factor*r['lower95']:+.6f}, {factor*r['upper95']:+.6f}] ({'điểm phần trăm' if factor==100 else 'tau-c'}); {interpretation}.")
    section('1. Phạm vi và kết luận từ lần chạy này',
            f"Nguồn: {evidence_label}. Báo cáo được dựng từ predictions, selection, CI và timing của đúng producer có SHA bên dưới; không đọc metric cũ để điền kết quả mới. Primary cố định là mean_single, seed11.\n\n"+'\n\n'.join(findings),main,
            ['variant','sugar_accuracy_pct','flickr_raw_tau_c','polaris_raw_tau_c','adapter_parameters'], '01_quality')
    section('2. Idea và phương pháp',
            'Backbone google/embeddinggemma-2 frozen, FP32; embedding chuẩn hóa 768 chiều. Dùng chung residual adapter cho ảnh và text: r = W_up tanh(W_down x); b = r / max(1, ||r||₂); A(x) = normalize(x + radius × b). W_up khởi tạo 0; không bias. Score = 2.5 × max(0, cosine(Au, Av)), không phải xác suất. Prefix text giữ nguyên “A photo depicts ”, gồm khoảng trắng cuối.\n\n'
            'Loss pairwise softplus((margin − positive_cos + negative_cos)/0.05), Sugar margin 0.02, human margin 0. Hai nhánh dùng equal domain budgets và preservation embedding/score. Mean dùng trung bình votes để xếp cặp trong cùng ảnh. Uncertainty nhân trọng số gap/(gap + SE_pos + SE_neg + 1e-6), singleton SE=0 là heuristic. AdamW lr=0.001, weight_decay=0.0001, batch=128, tối đa 15 epochs.\n\n'
            'Chọn config/epoch chỉ trên validation qua ba seed. Gain chuẩn hóa [0.01, 0.02, 0.02]; yêu cầu không âm ở cả ba domain, tối đa hóa gain yếu nhất; nếu không có config hợp lệ thì fallback base. Refit train+validation bằng median epoch. Test không dùng chọn config, seed hay primary.')
    section('3. Dữ liệu, split và kiểm tra toàn vẹn',
            f"Tổng {len(records)} records; {len(test)} test records. Overlap ảnh giữa các split: {overlaps}. Sugar: strict greater, ties tính sai. Human: Kendall tau-c trên individual votes, caption mean và within-image ranking tách riêng. Không dùng reference caption khi tính score.",counts,['domain','split','records','images','votes'])
    section('4. Validation và quá trình học',json.dumps({p:v['selection'] for p,v in selection.items()},ensure_ascii=False,indent=2),trials,
            ['phase','config_index','rank','radius','preservation','gain_sugar','gain_flickr','gain_polaris','eligible','best_epochs'],'06_training_selection')
    section('5. Khoảng tin cậy paired và so sánh uncertainty',
            f"Paired image-cluster bootstrap, {protocol['statistics']['bootstrap_replicates']} resamples; giữ các caption/vote của mỗi ảnh. Các CI chưa điều chỉnh nhiều phép thử và không bao gồm toàn bộ biến thiên do training. CI point estimates được đối chiếu lại với metrics; report không chạy bootstrap chất lượng mới. Uncertainty có config/epoch được chọn riêng nên chênh lệch hai phase không cô lập nhân quả của weighting.",ci,
            ['contrast','metric','delta','lower95','upper95','unit','replicates'],'02_paired_ci')
    section('6. Ablation',
            'Các ablation dùng rank32/radius0.03/preservation0.1 (trừ no_preservation), seed11, 15 epochs. Đây là control cố định, không đồng nhất với số epoch của primary đã chọn. Giữ mọi kết quả âm.',ablations,
            ['variant','sugar_accuracy_pct','flickr_raw_tau_c','polaris_raw_tau_c'],'03_ablation')
    section('7. Phân tích lỗi và metric theo domain',
            'Bảng dưới đếm cặp Sugar được sửa và cặp đang đúng bị làm sai, ghép trên cùng record. sugar_changed_pairs.csv chứa toàn bộ thay đổi với caption và margin. Global raw tau-c và within-image ranking đo hai khía cạnh khác nhau; một metric tăng không bảo đảm metric kia tăng.',categories,
            ['variant','category','pairs','base_accuracy_pct','accuracy_pct','delta_pp','fixed','regressed','net_fixed'],'04_domain_errors')
    section('8. Dao động theo seed',
            'Tất cả seed refit của lần chạy này được giữ, không chọn seed có test tốt nhất. Ba seed là diagnostics; không coi phạm vi giữa seed là confidence interval. Các control CPU năm seed của nghiên cứu trước không được gọi là kết quả mới.',seed_rows,
            ['phase','seed','sugar_accuracy_pct','flickr_raw_tau_c','polaris_raw_tau_c'],'05_seed_variation')
    if timings:
        added=next(r for r in timings if r['variant']=='mean_single' and r['batch']==1)
        lo,hi=added['paired_added_ci95_ms_call']
        cost_text=(f"GPU {receipt['gpu']}, dtype {receipt['dtype']}. B1 mean overhead paired median {added['paired_added_p50_ms_call']:+.4f} ms/call; CI95 [{lo:+.4f}, {hi:+.4f}]. "
                   'CI timing resample input groups, giữ cả hai rounds. Throughput effective = batch × 1000 / median(ms/call), khác aggregate throughput. Peak là allocated memory, không phải toàn bộ VRAM/reserved memory. Mẫu 128 ảnh gồm 43 Sugar/43 Flickr/42 Polaris, 864 calls; gồm preprocessing, encoder, normalize, head, score; loại disk IO/model download. Cached-head latency báo riêng, không dùng làm native latency.')
    else:cost_text='Chưa cung cấp timing128: không suy diễn chi phí GPU từ cached-head hoặc số liệu lịch sử. Chạy lại report sau khi có timing.'
    section('9. Trade-off chất lượng, latency và bộ nhớ',cost_text,main,
            ['variant','adapter_parameters','adapter_fp32_kib','b1_p50_ms_call','b1_p95_ms_call','b8_effective_throughput_pairs_s','b8_peak_allocated_mib'],
            '07_native_cost' if timings else None)
    if historical:
        section('10. Ngữ cảnh so sánh bảy backbone đã có',
                'Đây là số liệu lưu từ nghiên cứu Stage12 ngày 16/10/2026; các model khác không được train/đo lại trong notebook Gemma. Gemma FP32, các recipe khác FP16; session khác nhau. Đồ thị chỉ là ngữ cảnh chất lượng/chi phí, không chứng minh hiệu ứng nhân quả kiến trúc hay speedup cùng precision. Bảng historical_seven_backbones.csv giữ revision, device ; data fingerprints remain in private inputs.',historical,
                ['backbone','precision','gpu','sugar_accuracy_pct_up','flickr_raw_tau_c_up','polaris_raw_tau_c_up','representative_b1_p50_ms','representative_b8_effective_pairs_s','representative_peak_mib'],
                '08_historical_tradeoff')
    if control_summary:
        section('10b. Control năm seed đã hoàn tất trước đó (lịch sử)',
                'Control CPU gốc: cùng rank32/radius0.03/preservation0.1, cùng 14 epochs và paired initialization/batches cho hai loss, năm seed11/23/37/53/71. Bảng này được copy và kiểm SHA, không được train lại trong notebook GPU. Chỉ so CPU với CPU; không dùng khác biệt CPU/CUDA để quy cho weighting. Mean vẫn là primary. historical_multiseed_per_seed.csv giữ từng seed; CI seed-only không thay CI seed×image.',control_summary,
                ['metric','mean_value','uncertainty_value','delta','unit','positive_seeds','negative_seeds','seeds','crossed_lower95','crossed_upper95'])
        section('10c. CI seed × image cho control lịch sử',
                'Bootstrap crossed đồng thời seed và image cluster, giữ paired predictions và toàn bộ votes. Flickr/Polaris raw weighting effect âm trong control lịch sử này; không suy thành uncertainty luôn có hại ở mọi corpus. Sugar crossed CI nằm ở bảng aggregate trên.',control_ci,
                ['domain','metric','delta','lower95','upper95','replicates','training_seeds'])
    section('11. Giới hạn và hướng tiếp theo',
            'Đây là continuation exploratory trên corpus đã quan sát, không fresh holdout/SOTA hoặc phục hồi exact Stage9/10. Chưa có audit độc lập về overlap với pretraining. Không diễn giải CI chứa 0 thành bằng nhau; không gọi uncertainty tốt hơn chỉ vì một point estimate cao. Không thay primary theo test.\n\n'
            'Ưu tiên nghiên cứu tiếp: fresh holdout/domain shift; paired controls cùng config/epoch trên nhiều seed để cô lập weighting; control precision/hardware cho so sánh chi phí; các baseline còn thiếu theo protocol mới. Không tự đưa các phép mở rộng này vào kết luận đã hoàn thành.')
    audit={'complete':True,'evidence_label':evidence_label,'created_utc':datetime.now(timezone.utc).isoformat(),
           'core_manifest_sha256':sha(core/'manifest_sha256.json'),'records_sha256':sha(core/'records.json'),
           'source_sha256':sha(__file__),'metrics_recomputed':len(metrics),'test_records':len(test),
           'prediction_mapping_verified':True,'split_image_overlap':overlaps,'selection_test_used':False,
           'timing':timing_audit,'historical_context_separate':True,'new_inference_by_report':False,
           'new_training_by_report':False,'figures':figures,'report_formats':['md','self_contained_html']}
    section('12. Reproducibility và danh mục output',
            f"Core manifest SHA256: {audit['core_manifest_sha256']}\n\nRecords SHA256: {audit['records_sha256']}\n\nModel revision: {receipt['revision']}\n\nRuntime: {json.dumps(read(core/'runtime.json'),ensure_ascii=False)}\n\n"
            'report_audit.json ghi phạm vi kiểm chứng. protocol.json, selection.json, model_receipt.json, runtime.json và recomputed_metrics.json lưu thông tin tái chạy. CSV giữ đủ variants, domain metrics, paired CI, errors, validation, training, seed và timing. Figures có PNG/SVG. HTML nhúng toàn bộ hình để tải một file vẫn xem được. Core/timing giữ predictions, checkpoints và raw evidence trong cùng thư mục lượt chạy.')
    def value(v):
        if v is None:return '—'
        return f'{v:.6f}' if isinstance(v,float) else str(v)
    markdown=['# Báo cáo Gemma — end to end',f"Nguồn dữ liệu: `{evidence_label}`. Tạo lúc {audit['created_utc']}."]
    html_parts=['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gemma research report</title>',
                '<style>body{max-width:1280px;margin:32px auto;padding:0 22px;font:15px/1.6 system-ui;color:#192a3c}h1,h2{color:#006b7c}h2{margin-top:42px}p{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border:1px solid #d3dce5;text-align:left}th{background:#e8f0f5}tr:nth-child(even){background:#f6f8fa}.scroll{overflow:auto}img{max-width:100%;height:auto}nav a{margin-right:14px}@media print{body{font-size:10pt}h2{break-after:avoid}tr,img{break-inside:avoid}.scroll{overflow:visible}table{font-size:8pt}}</style><body><h1>Báo cáo Gemma — end to end</h1>',
                f"<p>Nguồn dữ liệu: {html.escape(evidence_label)}. Tạo lúc {audit['created_utc']}.</p><nav>"]
    html_parts.extend(f'<a href="#s{i}">{html.escape(s[0])}</a>' for i,s in enumerate(sections));html_parts.append('</nav>')
    for i,(title,text,rows,columns,figure) in enumerate(sections):
        markdown += ['\n## '+title+'\n',text]
        html_parts += [f'<h2 id="s{i}">{html.escape(title)}</h2>','<p>'+html.escape(text)+'</p>']
        if rows:
            columns=columns or list(rows[0]);markdown+=['\n| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
            html_parts.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr></thead><tbody>')
            for row in rows:
                vals=[value(row.get(k)) for k in columns]
                markdown.append('| '+' | '.join(v.replace('|','\\|').replace('\n',' ') for v in vals)+' |')
                html_parts.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
            html_parts.append('</tbody></table></div>')
        if figure:
            markdown.append(f'\n![{figure}](figures/{figure}.png)\n')
            encoded=base64.b64encode((figdir/(figure+'.png')).read_bytes()).decode('ascii')
            html_parts.append(f'<img alt="{figure}" src="data:image/png;base64,{encoded}">')
    html_parts.append('</body></html>')
    (output/'REPORT_VI.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    (output/'REPORT_VI.html').write_text('\n'.join(html_parts),encoding='utf-8')
    write(output/'report_audit.json',audit);seal(output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--timing');parser.add_argument('--evidence-label',default='completed_native_gpu_producer')
    args=parser.parse_args()
    print(build_report(args.core,args.output,args.timing,args.evidence_label))
