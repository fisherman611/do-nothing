import json,random,sys
root=sys.argv[1]  # .../train+public_test
rows=[json.loads(l) for l in open(f'{root}/train/train.jsonl')]
ids=sorted(r['id'] for r in rows); random.Random(42).shuffle(ids)
val=set(ids[:int(len(ids)*0.08)])
out=[]
for r in rows:
    sp='val' if r['id'] in val else 'train'
    out.append(dict(uid=r['id'],audio=r['audio'],text=r['audio_text'],sid=r['id'],kind='final',split=sp))
    for t in r['history']:
        if t['role']=='user':
            out.append(dict(uid=t['audio'].split('/')[-1][:-4],audio=t['audio'],text=t['audio_text'],sid=r['id'],kind='history',split=sp))
json.dump(out,open('modal/manifest.json','w'),ensure_ascii=False)
from collections import Counter;print(Counter((o['split'],o['kind']) for o in out))
