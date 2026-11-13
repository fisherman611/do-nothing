import json,sys,os
P="review340_decisions.json"
def save(items):
    d=json.load(open(P,encoding="utf-8")) if os.path.exists(P) else {}
    for it in items:
        i,t,c=it[:3]; d[i]=[t,c]+([it[3]] if len(it)>3 else [])
    json.dump(d,open(P,"w",encoding="utf-8"),ensure_ascii=False,indent=0); print("đã có",len(d),"quyết định")
