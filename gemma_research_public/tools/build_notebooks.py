"""Generate portable notebooks with embedded code, never embedded credentials."""
import base64
import io
import inspect
import textwrap
import zipfile
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT=Path(__file__).resolve().parents[1]


def build():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['src','vendor/stage11','report_context']:
            for p in sorted((ROOT/folder).glob('*')):
                if p.is_file() and p.suffix in ['.py','.json','.csv']:z.writestr(p.relative_to(ROOT).as_posix(),p.read_bytes())
        for name in ['PINNED.json','requirements_gpu.txt']:
            z.writestr(name,(ROOT/name).read_bytes())
    payload=base64.b64encode(buffer.getvalue()).decode('ascii')
    bootstrap=f'''import os, sys, io, base64, tempfile, zipfile
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
roots=[Path.cwd(),Path.cwd().parent,Path.cwd()/'gemma_research_standalone']
CODE_ROOT=next((p.resolve() for p in roots if (p/'src/bundle.py').is_file() and (p/'PINNED.json').is_file()),None)
if CODE_ROOT is None:
    CODE_ROOT=Path(tempfile.mkdtemp(prefix='gemma_standalone_code_'))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as archive:
        for member in archive.infolist():
            if not (CODE_ROOT/member.filename).resolve().is_relative_to(CODE_ROOT.resolve()):
                raise ValueError('Embedded source path escaped destination')
        archive.extractall(CODE_ROOT)
sys.path.insert(0,str(CODE_ROOT/'src'))
from bundle import verify_code
pins=verify_code()
if globals().get('RUN_FULL_GPU',False):
    from bundle import require_private_config
    require_private_config()
print('Frozen scientific source verified. Model:',pins['gemma_model']['repo'])
'''
    md=lambda text:nbf.v4.new_markdown_cell(inspect.cleandoc(text))
    def code(text):
        lines=text.strip('\n').splitlines()
        if len(lines)>1 and all(not line.strip() or line.startswith(' '*6) for line in lines[1:]):
            lines=[lines[0].lstrip()]+[line[6:] for line in lines[1:]]
        return nbf.v4.new_code_cell('\n'.join(lines).strip())
    def notebook(cells,path):
        doc=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.13'},'gemma_bundle':{'contains_credentials':False,'auto_submit':False}})
        nbf.validate(doc)
        for cell in cells:
            cell.id=hashlib.sha256((cell.cell_type+cell.source).encode()).hexdigest()[:16]
            if cell.cell_type=='code':compile(cell.source,str(path),'exec')
        path.parent.mkdir(parents=True,exist_ok=True);nbf.write(doc,path)
    cpu=[
      md('''# Gemma: replay và paired refit trên CPU

      Notebook này dùng **cache/checkpoint thật**, không tải model và không dùng GPU. Chạy bằng môi trường uv trong README. Nếu chạy trên Kaggle, attach assets cùng `ASSET_MANIFEST_SHA256.json`; code đã nhúng sẵn. Public notebook không nhúng dataset fingerprints; attach cấu hình riêng theo README.

      Idea: backbone frozen, residual head chung image/text, correction bounded. `s=2.5 max(0,cos(Au,Av))`. Primary mean đã khóa; uncertainty secondary. Phép refit dưới đây giữ cùng config/14 epochs cho hai loss, là control CPU sau test, không thay primary CUDA.
      '''),
      code(bootstrap),
      code(r'''from datetime import datetime
      from bundle import find_bundle, verify_assets
      ASSETS_ROOT = find_bundle()  # Có thể thay bằng Path('/đường/dẫn/gemma_research_standalone')
      OUTPUT_ROOT = (Path('/kaggle/working') if Path('/kaggle/working').exists() else ASSETS_ROOT/'runs') / ('cpu_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
      RUN_REFIT = True
      REFIT_SEEDS = (11,)  # Đổi thành (11,23,37,53,71) để chạy năm seed.
      verify_assets(ASSETS_ROOT)
      print('Assets:',ASSETS_ROOT,'\nOutput:',OUTPUT_ROOT)
      '''),
      md('''## 1. Tái tính mọi metric từ saved predictions

      Giữ exact ID mapping, individual votes và strict Sugar ties. Sau đó chấm lại base/mean/uncertainty bằng checkpoint trên cache CPU; ghi max absolute score difference với CUDA gốc. Không coi sai khác làm tròn rất nhỏ là training mới.
      '''),
      code(r'''from replay import replay
      metrics,audit = replay(ASSETS_ROOT, OUTPUT_ROOT/'replay',rescore=True)
      assert audit['saved_variants_recomputed']==15 and audit['saved_metrics_match']
      print(audit)
      from IPython.display import display, Markdown
      table='| Variant | Sugar % | Flickr raw tau-c | Polaris raw tau-c |\n|---|---:|---:|---:|\n'
      for name in ['base','mean_single','uncertainty_single']:
          m=metrics[name]
          table+=f"| {name} | {100*m['sugar']['accuracy']:.5f} | {m['flickr']['raw_vote_tau_c']:.6f} | {m['polaris']['raw_vote_tau_c']:.6f} |\n"
      display(Markdown(table))
      '''),
      code('''import matplotlib.pyplot as plt
      names=['base','mean_single','uncertainty_single']
      fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
      for ax,(domain,key,factor,label) in zip(axes,[('sugar','accuracy',100,'Sugar accuracy (%)'),('flickr','raw_vote_tau_c',1,'Flickr raw tau-c'),('polaris','raw_vote_tau_c',1,'Polaris raw tau-c')]):
          ax.plot(range(3),[factor*metrics[n][domain][key] for n in names],'o-')
          ax.set_xticks(range(3),['Base','Mean','Uncertainty'],rotation=15)
          ax.set_title(label);ax.grid(alpha=.2)
      fig.suptitle('Saved CUDA Gemma results - same verified test cohort; exploratory')
      plt.show()
      '''),
      md('''## 2. Fit lại paired heads trên CPU

      Hai phase cùng initialization, sampled batches, config mean và14 epochs; chỉ loss weighting khác. Cùng seed11 là chạy nhanh; năm seeds để xem variability. Nếu Torch2.14.1+cpu khớp runtime gốc, code kiểm score với control đã lưu. Runtime khác được ghi riêng; không gọi là exact CUDA reproduction.
      '''),
      code('''if RUN_REFIT:
          from refit import refit
          refit_metrics,receipt=refit(ASSETS_ROOT,OUTPUT_ROOT/'refit',REFIT_SEEDS)
          assert receipt['same_initialization_and_sampled_batches']
          print(receipt)
          display(Markdown('**Refit hoàn tất. Đây là CPU control, primary gốc không thay đổi.**'))
      else:
          print('Refit skipped; set RUN_REFIT=True to train the paired heads.')
      '''),
      md('''## 3. Kết quả control năm seed đã có

      Đây là các control CPU đã kiểm chứng của nghiên cứu gốc, khác phép so mean14/uncertainty11 epochs trên CUDA. Không chọn seed tốt nhất và không ensemble để gọi là mean effect. CI seed×ảnh cho raw tau-c nằm ở bảng riêng, không lấy seed-only CI thay thế.
      '''),
      code('''import csv
      with (ASSETS_ROOT/'assets/reference/multiseed_summary.csv').open(encoding='utf-8-sig',newline='') as f:
          control_rows=list(csv.DictReader(f))
      with (ASSETS_ROOT/'assets/reference/crossed_global_ci.csv').open(encoding='utf-8-sig',newline='') as f:
          crossed_rows=list(csv.DictReader(f))
      for row in control_rows:
          if row['metric'] in ['sugar_accuracy','flickr_raw_vote_tau_c','polaris_raw_vote_tau_c']:
              print(row['metric'],'uncertainty-minus-mean:',row['delta'],row['unit'])
      for row in crossed_rows:
          if row['metric']=='raw_vote_tau_c':print(row['domain'],'paired seed x image CI:',row['lower95'],row['upper95'])
      print('Next: read IDEA_METHOD_VI.md / RESULTS_VI.md, or use the separate GPU notebook for native re-encoding.')
      ''')]
    notebook(cpu,ROOT/'notebooks/01_Gemma_CPU_Replay_Refit.ipynb')
    gpu=[
      md('''# Gemma: public source, private inputs, full Kaggle GPU rerun

      Attach **native inputs và `gemma_private_runtime.json` dưới dạng private Kaggle inputs (xem README_VI.md)**, bật Internet và GPU T4. Code đã nhúng trong notebook; không cần repo cha hoặc cache kết quả cũ. ZIP và cây thư mục do Kaggle tự giải nén đều được nhận và kiểm manifest.

      **Trên Kaggle: attach data, chọn GPU T4, bật Internet rồi Run All.** Hai biến install/run mặc định bật khi ở Kaggle; laptop không chạy phần nặng. Full route: encode5817 ảnh/20065 texts, validation grid, refit ba seeds, test2000-replicate CI, bốn ablations, timing128 ảnh → CSV/PNG/SVG → báo cáo tiếng Việt HTML/Markdown → ZIP tải về. Không tự publish notebook hoặc thay kết quả đã lưu.

      Báo cáo phân biệt kết quả mới với ngữ cảnh bảy backbone ngày16/10/2026. Không huấn luyện lại bảy backbone. Model weights tải đúng revision/SHA từ Hugging Face; nếu model yêu cầu quyền truy cập, chấp nhận điều khoản và thêm Kaggle Secret `HF_TOKEN`.
      '''),
      code('''from pathlib import Path
      from datetime import datetime, timezone
      ON_KAGGLE = Path('/kaggle/working').is_dir()
      INSTALL_GPU_DEPENDENCIES = ON_KAGGLE
      RUN_FULL_GPU = ON_KAGGLE
      RUN_TIMING_128 = True
      NATIVE_DATA = None  # Auto-discover native_inputs.zip / expanded snapshot under /kaggle/input.
      RUN_NAME = 'gemma_e2e_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
      '''),
      code(bootstrap),
      code('''if RUN_FULL_GPU:
          import torch
          if not torch.cuda.is_available():
              raise RuntimeError('Kaggle Settings -> Accelerator -> GPU T4, then restart session and Run All.')
          print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'CUDA:',torch.version.cuda)
          print('One GPU is used. Recommended: T4 16GB; the laptop is not used for computation.')
          if ON_KAGGLE and not os.environ.get('HF_TOKEN'):
              try:
                  from kaggle_secrets import UserSecretsClient
                  token = UserSecretsClient().get_secret('HF_TOKEN')
                  if token: os.environ['HF_TOKEN']=token
                  del token
              except Exception:
                  pass  # Public access may work; never print secrets or exception contents.
      if INSTALL_GPU_DEPENDENCIES:
          import subprocess
          subprocess.run([sys.executable,'-m','pip','install','--no-input','--retries','2','--timeout','30','-r',str(CODE_ROOT/'requirements_gpu.txt')],check=True,timeout=600)
      else:
          print('Dependency installation disabled. Enable it before a full GPU rerun if the pinned stack is not installed.')
      '''),
      md('''## Encode → validation-only selection → refit → test → CI/ablation → timing → report

      Native Gemma FP32 cố định; cache/head FP32. Source `learning.py`/`run.py` giữ nguyên byte. Wrapper chỉ inject explicit FP32 Encoder và quản lý input/output. Cùng protocol không bảo đảm bitwise-identical result ở GPU/runtime khác; mọi model weight SHA, recipe và runtime được ghi receipt.
      '''),
      code('''if RUN_FULL_GPU:
          import importlib.metadata as metadata
          required={'numpy':'2.3.5','scipy':'1.17.0','transformers':'5.19.0','sentence-transformers':'6.1.0','huggingface_hub':'1.33.0'}
          for package,version in required.items():
              assert metadata.version(package)==version, f'Enable INSTALL_GPU_DEPENDENCIES for {package}=={version}'
          import subprocess
          OUTPUT=(Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()/'runs')/RUN_NAME
          command=[sys.executable,'-u',str(CODE_ROOT/'src/run_native.py'),'--output',str(OUTPUT)]
          if NATIVE_DATA:command+=['--data',str(NATIVE_DATA)]
          if not RUN_TIMING_128:command+=['--skip-timing']
          # Fresh interpreter uses installed package versions even if Kaggle preloaded NumPy.
          process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
          for line in process.stdout:print(line,end='',flush=True)
          if process.wait()!=0:raise RuntimeError('GPU pipeline failed; see streamed logs and execution_status.json. Completed evidence is retained.')
          print('Full Gemma end-to-end run complete:',OUTPUT)
      else:
          print('Full native run disabled. Set INSTALL_GPU_DEPENDENCIES=True and RUN_FULL_GPU=True, attach inputs, then Run All on a GPU.')
      '''),
      code('''if RUN_FULL_GPU:
          from bundle import read
          from common import verify
          verify(OUTPUT)
          assert read(OUTPUT/'execution_status.json')['complete']
          assert read(OUTPUT/'core/status.json')['complete']
          m=read(OUTPUT/'core/gemma/metrics.json')
          for name in ['base','mean_single','uncertainty_single']:print(name,m[name])
          if RUN_TIMING_128:
              timing=OUTPUT/'timing128'
              verify(timing)
              assert read(timing/'bootstrap_status.json')['complete']
              assert len(read(timing/'timing_results.json'))==6
              assert len(read(timing/'timing_samples.json'))==864
              print('128-image native timing verified: six rows / 864 calls.')
          verify(OUTPUT/'report')
          audit=read(OUTPUT/'report/report_audit.json')
          assert audit['complete'] and audit['evidence_label']=='new_native_gpu_run'
          print('Audited report:',OUTPUT/'report/REPORT_VI.html')
      '''),
      md('''## Báo cáo, biểu đồ và tải toàn bộ kết quả

      Mỗi lượt chạy tạo một folder riêng: `core/`, `timing128/`, `report/`, code snapshot, execution receipt và manifest SHA256. ZIP cạnh folder chứa toàn bộ kết quả, gồm cache mới/checkpoints/predictions; ảnh input và pretrained weights không được copy vào ZIP. HTML tự chứa hình, có thể xem offline hoặc dùng Print → Save as PDF.
      '''),
      code('''if RUN_FULL_GPU:
          from IPython.display import display, Markdown, Image, FileLink
          report_lines=(OUTPUT/'report/REPORT_VI.md').read_text(encoding='utf-8').splitlines()
          display(Markdown(chr(10).join(line for line in report_lines if not line.startswith('!['))))
          for path in sorted((OUTPUT/'report/figures').glob('*.png')):
              display(Image(filename=str(path)))
          print('Download from Kaggle Output / Save Version:')
          for path in [OUTPUT/'report/REPORT_VI.html',OUTPUT/'report/REPORT_VI.md',OUTPUT.with_name(OUTPUT.name+'_results.zip')]:
              display(FileLink(str(path)))
      '''),
      md('''## Giữ output và diễn giải

      Save Version (Save & Run All) để Kaggle giữ output ngay cả khi laptop tắt: báo cáo, biểu đồ, CSV, ZIP, records/mapping, native cache, checkpoints, metrics, paired CI, validation trials, training histories và raw timing. Mean seed11 vẫn primary; uncertainty/ensemble/ablation không được dùng để chọn lại theo test.

      Có thể rebuild public inputs bằng `vendor/stage11/prepare_data.py` khi cần protocol mới/khôi phục; snapshot đính kèm là đường chạy gốc đã kiểm SHA. Số liệu hiện tại exploratory, không fresh holdout/SOTA hoặc exact Stage9/10 recovery.
      ''')]
    notebook(gpu,ROOT/'notebooks/02_Gemma_Kaggle_Full_Rerun.ipynb')
    print('PORTABLE_NOTEBOOKS_BUILT')


if __name__=='__main__':build()
