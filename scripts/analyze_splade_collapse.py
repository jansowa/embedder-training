"""Read-only parameter transplants and initial-loss diagnostics for SPLADE collapse."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original', default='OPI-PIB/pl-ModernBERT-512-base')
    parser.add_argument('--checkpoint', default='models/modernbert-probe-m-s00625')
    parser.add_argument('--final', default='models/modernbert-probe-m-s05000')
    parser.add_argument('--rows', default='datasets/judged/msmarco-clean-20000.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from safetensors.torch import load_file
    from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    report={'transplants':[], 'loss_analysis':{}, 'parameter_changes':{}}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))

    tokenizer=AutoTokenizer.from_pretrained(args.original,local_files_only=True)
    model=AutoModelForMaskedLM.from_pretrained(args.original,local_files_only=True,torch_dtype=torch.float32)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    collapsed_model=AutoModelForMaskedLM.from_pretrained(args.checkpoint,local_files_only=True,torch_dtype=torch.float32)
    collapsed={k:v.detach().clone() for k,v in collapsed_model.state_dict().items()}
    del collapsed_model
    final=load_file(str(Path(args.final)/'model.safetensors'),device='cpu')
    first=load_file(str(Path(args.checkpoint)/'model.safetensors'),device='cpu')
    changed=[name for name in first if name in final and not torch.equal(first[name],final[name])]
    report['parameter_changes']['between_625_and_5000']={'different_tensors':changed,'compared_tensors':len(set(first)&set(final))}
    del first, final
    groups={}
    for name, before in original.items():
        after=collapsed[name]
        if not before.is_floating_point(): continue
        group='head' if name.startswith(('head.','decoder.bias')) else '.'.join(name.split('.')[:3])
        item=groups.setdefault(group,{'delta_sq':0.0,'original_sq':0.0,'max_abs_delta':0.0})
        diff=after.float()-before.float()
        item['delta_sq']+=float((diff*diff).sum())
        item['original_sq']+=float((before.float()*before.float()).sum())
        item['max_abs_delta']=max(item['max_abs_delta'],float(diff.abs().max()))
    for group,item in groups.items(): item['relative_l2_delta']=math.sqrt(item['delta_sq']/max(item['original_sq'],1e-30))
    report['parameter_changes']['groups']=groups
    model=model.to('cuda').eval()
    texts=['Warszawa jest stolicą Polski i leży nad Wisłą.','Jakie miasto jest stolicą Polski?',
           'Podatek dochodowy jest naliczany od uzyskanego dochodu.',
           'Woda wrze w temperaturze stu stopni Celsjusza przy normalnym ciśnieniu.']
    encoded=tokenizer(texts,padding=True,return_tensors='pt').to('cuda')
    def head_parameter(name): return name.startswith('head.') or name=='decoder.bias'
    def probe(name,state):
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            logits=model(**encoded).logits
            pooled=(torch.log1p(logits.relu())*encoded['attention_mask'].unsqueeze(-1)).amax(1)
            item={'variant':name,'active_dims':(torch.round(pooled[:,:len(tokenizer)])*100>0).sum(1).cpu().tolist(),
                  'quantized_dims':(torch.round(pooled[:,:len(tokenizer)]*100)>0).sum(1).cpu().tolist(),
                  'max_logit':float(logits.max()),'min_logit':float(logits.min())}
        report['transplants'].append(item)
        print(json.dumps(item),flush=True)
        save()
    probe('original',original)
    probe('collapsed',collapsed)
    probe('original_backbone_trained_head',{k:collapsed[k] if head_parameter(k) else v for k,v in original.items()})
    probe('trained_backbone_original_head',{k:original[k] if head_parameter(k) else v for k,v in collapsed.items()})
    for low,high in [(0,10),(11,21)]:
        def selected(name):
            parts=name.split('.')
            return len(parts)>2 and parts[0]=='model' and parts[1]=='layers' and parts[2].isdigit() and low<=int(parts[2])<=high
        probe(f'collapsed_restore_layers_{low}_{high}',{k:original[k] if selected(k) else v for k,v in collapsed.items()})

    # Compute objective signals on real initial embeddings. No optimizer steps
    # are made; gradients stop at detached embedding leaves, not model weights.
    model.load_state_dict(original)
    model.train()
    rows=[]
    with open(args.rows) as handle:
        for line in handle:
            row=json.loads(line)
            if row.get('pos') and len(row.get('neg',[]))>=6: rows.append(row)
    rows=random.Random(42).sample(rows,8)
    columns=[[row['query'] for row in rows],[row['pos'][0] for row in rows]]
    columns.extend([[row['neg'][n] for row in rows] for n in range(6)])
    vectors=[]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    for texts_column in columns:
        batch=tokenizer(texts_column,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(**batch).logits
            pooled=(torch.log1p(logits.relu())*batch['attention_mask'].unsqueeze(-1)).amax(1)
        vectors.append(pooled.detach().requires_grad_(True))
        del batch,logits,pooled
    ranking=SparseMultipleNegativesRankingLoss(model,scale=1.0)
    wrapper=SpladeLoss(model,loss=ranking,document_regularizer_weight=1.0,query_regularizer_weight=1.0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        base=ranking.compute_loss_from_embeddings(vectors,None)
        doc=wrapper.document_regularizer.compute_loss_from_embeddings(torch.cat(vectors[1:]))
        query=wrapper.query_regularizer.compute_loss_from_embeddings(vectors[0])
        scores=vectors[0]@torch.cat(vectors[1:]).T
    gradients={}
    for name,loss in [('ranking',base),('document_regularizer_unweighted',doc),('query_regularizer_unweighted',query)]:
        grads=torch.autograd.grad(loss,vectors,retain_graph=True,allow_unused=True)
        norm=math.sqrt(sum(float((g.float()**2).sum()) for g in grads if g is not None))
        radial=sum(float((g.float()*v.float()).sum()) for g,v in zip(grads,vectors) if g is not None)
        gradients[name]={'loss':float(loss),'embedding_gradient_l2':norm,'uniform_expansion_directional_derivative':radial}
    report['loss_analysis']={'micro_batch':8,'candidate_documents':56,
        'score_min':float(scores.min()),'score_max':float(scores.max()),
        'score_std':float(scores.float().std()),'gradient_components':gradients,
        'document_weight_at_step25':0.003*(25/1000)**2,
        'document_weight_at_step250':0.003*(250/1000)**2,
        'uniform_shrink_controls':[],'score_scale_controls':[]}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for factor in [1.0,0.5,0.1,0.01,0.0]:
            loss=ranking.compute_loss_from_embeddings([v*factor for v in vectors],None)
            report['loss_analysis']['uniform_shrink_controls'].append({'embedding_multiplier':factor,'ranking_loss':float(loss)})
        for scale in [1.0,0.01,0.001,0.0001]:
            ranking.scale=scale
            loss=ranking.compute_loss_from_embeddings(vectors,None)
            report['loss_analysis']['score_scale_controls'].append({'scale':scale,'ranking_loss':float(loss)})
    print(json.dumps(report['loss_analysis'],indent=2),flush=True)
    save()
    print(f'Report: {args.output}',flush=True)


if __name__=='__main__': main()
