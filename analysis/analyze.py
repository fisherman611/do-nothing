import csv, html, json, statistics, wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'train+public_test'
OUT = Path(__file__).resolve().parent
splits = {}
for split in ('train', 'public_test'):
    rows = [json.loads(x) for x in (ROOT / split / f'{split}.jsonl').read_text().splitlines() if x.strip()]
    data = {'rows': rows, 'main': [], 'history': [], 'words': [], 'chars': [], 'calls': Counter(), 'samples': Counter(), 'available': Counter(), 'histcalls': Counter(), 'call_counts': Counter(), 'distinct_counts': Counter(), 'errors': [], 'rates': Counter()}
    for row in rows:
        for typ, item in [('main', row)] + [('history', h) for h in row.get('history', []) if h.get('role') == 'user' and h.get('audio')]:
            try:
                with wave.open(str(ROOT / split / item['audio']), 'rb') as f:
                    data[typ].append(f.getnframes() / f.getframerate())
                    data['rates'][f'{f.getframerate()} Hz / {f.getnchannels()} channel'] += 1
            except Exception as e:
                data['errors'].append([row['id'], item.get('audio'), str(e)])
        if 'audio_text' in row:
            data['words'].append(len(row['audio_text'].split()))
            data['chars'].append(len(row['audio_text']))
        for app in row['context']['installed_apps'].values():
            data['available'].update(set(app.get('tools', [])))
        for h in row.get('history', []):
            data['histcalls'].update(c['name'] for c in h.get('tool_calls', []))
        if 'label' in row:
            calls = [c['name'] for c in row['label']['tool_calls']]
            data['calls'].update(calls); data['samples'].update(set(calls))
            data['call_counts'][len(calls)] += 1; data['distinct_counts'][len(set(calls))] += 1
    splits[split] = data

def percentile(xs, p):
    xs = sorted(xs); pos = (len(xs)-1)*p; i = int(pos)
    return xs[i] + (xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)
def summary(xs):
    return {'n':len(xs), 'total':sum(xs), 'min':min(xs), 'mean':statistics.mean(xs), 'p50':percentile(xs,.5), 'p90':percentile(xs,.9), 'p95':percentile(xs,.95), 'p99':percentile(xs,.99), 'max':max(xs)}
def bins(xs, edges):
    counts = [0]*(len(edges)-1)
    for x in xs:
        for i,(a,b) in enumerate(zip(edges,edges[1:])):
            if a <= x < b: counts[i] += 1; break
    return [(f'{a}–{b}' if b != float('inf') else f'≥{a}', n) for (a,b),n in zip(zip(edges,edges[1:]), counts)]
def write_csv(name, header, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def bars(title, items, denom, unit='mẫu'):
    m=max((n for _,n in items),default=1) or 1
    return f'<h3>{html.escape(title)}</h3><div class="bars">' + ''.join(f'<div class="barrow"><span>{html.escape(str(label))}</span><div class="track"><div class="fill" style="width:{100*n/m:.2f}%"></div></div><b>{n:,} ({100*n/denom:.1f}%)</b></div>' for label,n in items) + f'</div><small>Mẫu số: {denom:,} {unit}.</small>'

report = ['''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Phân tích ZAIC train và public test</title><style>body{font:15px system-ui,sans-serif;max-width:1150px;margin:36px auto;padding:0 20px;color:#172332;line-height:1.5}h1,h2{color:#133f5d}h2{border-bottom:1px solid #ccd8df;padding-bottom:8px;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid #dce4e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf2f5}.barrow{display:grid;grid-template-columns:105px 1fr 140px;gap:10px;align-items:center;margin:4px 0}.barrow b{font-size:13px}.track{background:#edf2f4;height:17px}.fill{height:100%;background:#16867e}small{color:#526675}.note{background:#fff3dc;padding:12px;border-left:4px solid #d4992c}details{margin:20px 0}code{background:#eef3f4;padding:2px}</style><h1>Phân tích ZAIC: train và public test</h1><p>Dữ liệu v3.2.0; mỗi dòng JSONL là một mẫu. Audio chính là yêu cầu cuối cần xử lý; audio history là các lượt user trước đó. Chỉ tính audio được JSONL tham chiếu. Percentile nội suy tuyến tính.</p>''']
report.append('<h2>1. Audio</h2>')
sumrows=[]
for split,d in splits.items():
    for typ in ('main','history'):
        if d[typ]:
            s=summary(d[typ]); sumrows.append([split,typ,s['n'],f"{s['total']/3600:.2f}"]+[f"{s[k]:.2f}" for k in ('min','mean','p50','p90','p95','p99','max')])
report.append(table(['Tập','Audio','Số file','Tổng giờ','Min (s)','TB (s)','P50','P90','P95','P99','Max'],sumrows))
for split,d in splits.items():
    for typ in ('main','history'):
        if d[typ]: report.append(bars(f'{split}: audio {typ}, giây',bins(d[typ],[0,5,10,15,20,30,45,60,90,120,float('inf')]),len(d[typ]),'file'))
report.append('<h2>2. Transcript train</h2><p>Chỉ tính <code>audio_text</code> của yêu cầu cuối. Số từ là số đơn vị cách nhau bởi khoảng trắng, không phải số token của mô hình; ký tự gồm cả dấu cách và dấu câu.</p>')
d=splits['train']; textrows=[]
for name,xs in [('Từ',d['words']),('Ký tự',d['chars'])]:
    s=summary(xs); textrows.append([name,s['n'],int(s['total'])]+[f"{s[k]:.1f}" for k in ('min','mean','p50','p90','p95','p99','max')])
report.append(table(['Đơn vị','Transcript','Tổng','Min','TB','P50','P90','P95','P99','Max'],textrows))
report.append(bars('Số từ / transcript',bins(d['words'],[0,10,20,30,40,50,75,100,150,float('inf')]),len(d['words'])))
report.append(bars('Số ký tự / transcript',bins(d['chars'],[0,50,100,150,200,300,400,600,800,float('inf')]),len(d['chars'])))
report.append('<h2>3. Tool call</h2><p class="note">Public test không có label của yêu cầu cuối: không thể suy ra số single/multi-tool hay tần suất tool thực tế. Installed tools chỉ cho biết tool được phép dùng, không phải tool được gọi. Tool call trong history là các lượt trước, không phải nhãn test.</p><p>Train: single = đúng 1 call; multi = từ 2 call trở lên, kể cả gọi lặp cùng một tool. Bảng tool đếm cả số lần gọi và số mẫu có ít nhất một lần gọi.</p>')
n=len(d['rows']); cc=d['call_counts']
report.append(bars('Số call của yêu cầu cuối — train',[(str(k),v) for k,v in sorted(cc.items())],n))
report.append(bars('Số tool khác nhau / mẫu — train',[(str(k),v) for k,v in sorted(d['distinct_counts'].items())],n))
report.append('<h3>Toàn bộ tool trong nhãn train</h3>')
tools=sorted(set(d['calls'])|set(d['available'])|set(splits['public_test']['available']))
tr=[]
for t in sorted(tools,key=lambda t:(-d['calls'][t],t)):
    if d['calls'][t]: tr.append([t,d['calls'][t],d['samples'][t],f"{100*d['samples'][t]/n:.1f}%"])
report.append(table(['Tool','Số call','Số mẫu','% mẫu train'],tr))
report.append('<h3>Tool có sẵn theo mẫu</h3><p>Số mẫu có tool được cài; một mẫu có nhiều tool. Đây không phải phân phối tool được gọi.</p>')
report.append(table(['Tool','Train: số mẫu','Train %','Public test: số mẫu','Public test %'],[[t,d['available'][t],f"{100*d['available'][t]/n:.1f}%",splits['public_test']['available'][t],f"{100*splits['public_test']['available'][t]/len(splits['public_test']['rows']):.1f}%"] for t in sorted(tools,key=lambda t:(-splits['public_test']['available'][t],t))]))
report.append('<h3>Tool call trong history</h3>')
report.append(table(['Tập','Số call trong history','Số loại tool'],[[s,sum(x['histcalls'].values()),len(x['histcalls'])] for s,x in splits.items()]))
report.append('<p>Phân phối đầy đủ được lưu tại <code>tool_distribution.csv</code>; mã tái lập: <code>analyze.py</code>.</p></html>')
(OUT/'report.html').write_text('\n'.join(report),encoding='utf-8')
write_csv('tool_distribution.csv',['tool','train_label_calls','train_label_samples','train_available_samples','public_test_available_samples','train_history_calls','public_test_history_calls'],[[t,d['calls'][t],d['samples'][t],d['available'][t],splits['public_test']['available'][t],d['histcalls'][t],splits['public_test']['histcalls'][t]] for t in tools])
result={}
for split,x in splits.items():
    result[split]={'samples':len(x['rows']),'audio_main':summary(x['main']),'audio_history':summary(x['history']) if x['history'] else None,'audio_errors':x['errors'],'audio_formats':dict(x['rates']),'history_calls':dict(x['histcalls']),'call_counts':dict(x['call_counts']),'distinct_counts':dict(x['distinct_counts']),'top_calls':x['calls'].most_common(20),'transcript_words':summary(x['words']) if x['words'] else None,'transcript_chars':summary(x['chars']) if x['chars'] else None}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
