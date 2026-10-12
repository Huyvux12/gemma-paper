"""A standalone, labeled review gallery; diagnostic selection, not a benchmark."""
import base64,html,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from analyze import load,G,L,S,D,ROOT

def main():
    rows,scores,_=load('sugar');hit={k:v['positive']>v['negative'] for k,v in scores.items()};cards=[];selected=[]
    groups=[('Gemma đúng, CLIP L/14 sai',hit[G]&~hit[L]),('Gemma sai, CLIP L/14 đúng',~hit[G]&hit[L]),
      ('Cả Gemma và CLIP L/14 đều sai',~hit[G]&~hit[L]),('Gemma sai, SPECS đúng',~hit[G]&hit[S]),
      ('Gemma sai, SigLIP 2 đúng',~hit[G]&hit[D])]
    for title,mask in groups:
        used=set();indices=[]
        for i,r in enumerate(rows):
            if not r['previously_seen'] and mask[i] and r['category'] not in used:
                used.add(r['category']);indices.append(i)
        cards.append('<h2>'+html.escape(title)+f' ({len(indices)} ví dụ)</h2><div class="grid">')
        for i in indices:
            r=rows[i];selected.append({'group':title,'id':r['id'],'filename':r['filename']})
            with Image.open(ROOT/'images'/r['filename']) as im:
                im=im.convert('RGB');im.thumbnail((500,350));buf=io.BytesIO();im.save(buf,format='JPEG',quality=85)
            url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
            table='<table><tr><th>Method</th><th>Đúng</th><th>Sai</th><th>Chọn</th></tr>'
            for k,label in [(G,'Gemma 768'),(L,'CLIP L/14'),(S,'SPECS'),(D,'SigLIP 2')]:
                v=scores[k];table+=f'<tr><td>{label}</td><td>{v["positive"][i]:.4f}</td><td>{v["negative"][i]:.4f}</td><td>{"đúng" if hit[k][i] else "sai"}</td></tr>'
            table+='</table>'
            cards.append(f'<article><img src="{url}" alt="{html.escape(r["filename"])}"><p class="id">{html.escape(r["id"])} · {r["category"]}</p><p><b>Caption đúng theo bộ dữ liệu:</b> {html.escape(r["caption"])}</p><p><b>Caption sai theo bộ dữ liệu:</b> {html.escape(r["negative_caption"])}</p>{table}</article>')
        cards.append('</div>')
    text='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kiểm tra lỗi caption</title><style>body{font:16px/1.55 system-ui,sans-serif;background:#f6f7fb;color:#182437;max-width:1400px;margin:auto;padding:28px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}article{background:white;padding:20px;border:1px solid #dae0eb;border-radius:12px}img{display:block;max-width:100%;height:250px;object-fit:contain;margin:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #e4e7ee;text-align:left}.id{color:#54687d;font-size:14px}h2{margin-top:40px}header{max-width:900px}</style><header><h1>Kiểm tra các trường hợp sai và bất đồng</h1><p>Chỉ dùng ảnh chưa xuất hiện ở các lượt trước. Mỗi nhóm lấy ví dụ đầu tiên theo thứ tự manifest cho mỗi loại lỗi, tối đa 7 ví dụ. Đây là tập minh họa có chủ đích, không dùng để tính tần suất lỗi hoặc chọn prefix.</p><p>Những nhãn “đúng/sai” ở đây là nhãn của SugarCrepe. Bạn có thể kiểm tra lại nếu ảnh hoặc mô tả có chi tiết mơ hồ. So sánh hai caption <b>trong cùng một method</b>; các thang điểm giữa method khác nhau.</p></header>'''+''.join(cards)+'</html>'
    (ROOT/'error_gallery.html').write_text(text);(ROOT.parent/'EmbeddingGemma2_Error_Gallery.html').write_text(text)
    (ROOT/'gallery_selection.json').write_text(json.dumps({'selection':'first unseen error per category, deterministic manifest order','examples':selected},indent=2))
    print('Gallery',len(selected),'cards',flush=True)

if __name__=='__main__':main()
