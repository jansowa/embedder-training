import gc,json
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer
root=Path(__file__).resolve().parents[3]
p=root/'models/night3-paired-a-s04306'
passages=root/'third_party/pirb/data/scifact-pl/passages/passages.jsonl'
with passages.open() as f:texts=[json.loads(next(f))['contents'] for _ in range(8)]
tok=AutoTokenizer.from_pretrained(p)
batch=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
vectors={};records={}
for name,dtype,kwargs in [('old_fp16',torch.float16,{}),('fixed_fp16',torch.float16,{'device_map':'cpu'}),('fixed_fp32',torch.float32,{'device_map':'cpu'})]:
    m=AutoModelForMaskedLM.from_pretrained(p,torch_dtype=dtype,**kwargs).to('cuda').eval()
    with torch.inference_mode():v=(torch.log1p(torch.relu(m(**batch).logits))*batch['attention_mask'].unsqueeze(-1)).amax(1).float().cpu()
    vectors[name]=v
    records[name]={'bias_mean':float(m.get_output_embeddings().bias.float().mean()),'dims':(torch.round(v*100)>0).sum(1).tolist()}
    del m;gc.collect();torch.cuda.empty_cache()
records['old_vs_fixed_fp16_exact']=torch.equal(vectors['old_fp16'],vectors['fixed_fp16'])
records['fp32_fp16_max_abs_delta']=float((vectors['fixed_fp32']-vectors['fixed_fp16']).abs().max())
print(json.dumps(records,indent=2),flush=True)
(root/'experiments/polish-splade-2026-10/results/data/night3-bias-vectors.json').write_text(json.dumps(records,indent=2)+'\n')
