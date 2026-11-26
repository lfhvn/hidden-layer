// Run: node selves/soup/ancestry.cjs
// Lexical ancestry test for the random-draw runs. Ties are worth half.
// The planted control shuffles word order and therefore tests lexical detection only.
const fs = require("node:fs");
const path = require("node:path");
const load = name => fs.readFileSync(path.join(__dirname, "results", name), "utf8");
const text = row => console.log(JSON.stringify(row));

const tokens=s=>s.toLowerCase().match(/[\p{L}\p{N}_]+/gu)??[];
function hash(s){let h=2166136261;for(const c of s)h=Math.imul(h^c.charCodeAt(0),16777619);return h>>>0;}
function counts(w){const m=new Map();for(const t of w)m.set(t,(m.get(t)??0)+1);return m;}
function cosine(a,b){let aa=0,bb=0,ab=0;for(const [t,n] of a){aa+=n*n;ab+=n*(b.get(t)??0)}for(const n of b.values())bb+=n*n;return aa&&bb?ab/Math.sqrt(aa*bb):0;}
function grams(w){const m=new Map();for(let i=0;i+5<=w.length;i++){const k=w.slice(i,i+5).join(" ");const v=m.get(k)??[];v.push(i);m.set(k,v)}return m;}
function mask(w,g,other,out=new Set()){for(const [k,starts] of g){if(other.has(k))for(const i of starts)for(let j=i;j<i+5;j++)out.add(j)}return out;}
function compare(c,a,b){const cm=mask(c.w,c.g,a.g);mask(c.w,c.g,b.g,cm);const filtered=(x,m)=>counts(x.w.filter((_,i)=>!m.has(i)));const ct=filtered(c,cm),x=cosine(ct,filtered(a,mask(a.w,a.g,c.g))),y=cosine(ct,filtered(b,mask(b.w,b.g,c.g)));return Math.abs(x-y)<1e-12?0.5:Number(x>y);}
function doc(s,parents=[],pool=[]){const w=tokens(s);return {w,g:grams(w),bag:counts(w),parents,pool};}
const seeds=JSON.parse(load("init.json"));
for(const model of ["base","instruct"]){
const rows=load(model+".jsonl").trim().split("\n").map(JSON.parse).filter(r=>r.cond==="random"&&r.text);
for(let seed=0;seed<4;seed++){
const docs=new Map(seeds[seed].map((s,i)=>[i,doc(s)])), founders=[...docs.keys()],pool=new Set(founders),out={};
for(const r of rows.filter(r=>r.seed===seed)){
const c=doc(r.text,r.parents,[...pool]);docs.set(r.id,c);pool.delete(r.removed);pool.add(r.id);
let generation=[r.id],exclude=new Set();
for(let d=1;d<=4;d++){
generation=[...new Set(generation.flatMap(i=>docs.get(i).parents))].sort((a,b)=>a-b);for(const i of generation)exclude.add(i);
if(d===3||!generation.length)continue;
const key=[model,seed,r.id,d].join("/"), aid=generation[hash(key+"/a")%generation.length],a=docs.get(aid);
const shortlist=(a.pool.length?a.pool:founders).filter(i=>i!==aid&&!exclude.has(i)).map(i=>{const b=docs.get(i);return {i,dist:1-cosine(a.bag,b.bag)+Math.abs(Math.log((1+a.w.length)/(1+b.w.length)))}}).sort((a,b)=>a.dist-b.dist||a.i-b.i).slice(0,5);
if(!shortlist.length)continue;
const b=docs.get(shortlist[hash(key+"/b")%shortlist.length].i),win=compare(c,a,b),o=out[d]??={n:0,win:0,ties:0,perm:0,exact:0,controlN:0,controlWin:0};
o.n++;o.win+=win;o.ties+=+(win===0.5);o.perm+=hash(key+"/perm")&1?1-win:win;o.exact+=+[...c.g.keys()].some(g=>a.g.has(g));
if(a.w.length>=8){const p=doc(a.w.map((w,i)=>({w,h:hash(key+"/plant/"+i)})).sort((x,y)=>x.h-y.h).map(x=>x.w).join(" "));if(![...p.g.keys()].some(g=>a.g.has(g))){o.controlN++;o.controlWin+=compare(p,a,b);}}
}
}
text({model,seed,out});
}
}
