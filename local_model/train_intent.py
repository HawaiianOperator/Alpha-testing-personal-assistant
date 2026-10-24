"""Train a small text intent classifier from random weights; no pretrained model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .text_features import DIMENSIONS, LABELS, features
from .intent_dataset import EVALUATION

def train(data, outdir, epochs=400, seed=7):
    records=[json.loads(line) for line in Path(data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or set(row["label"] for row in records) != set(LABELS):
        raise ValueError("Provide labeled search, compute, and reject examples.")
    for row in records:
        if not isinstance(row["text"],str) or not row["text"].strip(): raise ValueError("Empty/nontext training input.")
    evaluation_text={text.casefold().strip() for text,_ in EVALUATION}
    if any(row["text"].casefold().strip() in evaluation_text for row in records):
        raise ValueError("Fixed evaluation prompt appears in training; remove overlap or define a new independent evaluation set.")
    x=np.zeros((len(records),DIMENSIONS),dtype=np.float32)
    for i,row in enumerate(records):
        for k,v in features(row["text"]).items(): x[i,k]=v
    y=np.array([LABELS.index(row["label"]) for row in records])
    # Reweight classes so rejection does not dominate solely by example count.
    sample_weight=len(y)/(len(LABELS)*np.bincount(y)[y])
    rng=np.random.default_rng(seed)
    w=rng.normal(0,0.01,(len(LABELS),DIMENSIONS)).astype(np.float32)
    b=np.zeros(len(LABELS),dtype=np.float32)
    mw=np.zeros_like(w); vw=np.zeros_like(w); mb=np.zeros_like(b); vb=np.zeros_like(b)
    started=time.monotonic()
    for epoch in range(1,epochs+1):
        scores=x@w.T+b; scores-=scores.max(1,keepdims=True)
        p=np.exp(scores); p/=p.sum(1,keepdims=True)
        loss=float(np.mean(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12))*sample_weight))
        p[np.arange(len(y)),y]-=1
        p*=sample_weight[:,None]/len(y)
        gw=p.T@x+0.0001*w; gb=p.sum(0)
        for param,grad,m,v in ((w,gw,mw,vw),(b,gb,mb,vb)):
            m*=0.9; m+=0.1*grad; v*=0.999; v+=0.001*grad*grad
            param-=0.03*(m/(1-0.9**epoch))/(np.sqrt(v/(1-0.999**epoch))+1e-8)
        if epoch==1 or epoch%100==0: print(f"epoch {epoch}: loss={loss:.4f}",flush=True)
    output=Path(outdir); output.mkdir(parents=True,exist_ok=True)
    model={"format":"hashed-ngram-softmax-v1", "labels":list(LABELS), "dimensions":DIMENSIONS,
        "weights":w.tolist(),"bias":b.tolist(),"min_probability":0.8,"min_margin":0.35,
        "initialization":"random; no pretrained weights","seed":seed,
        "limitations":"Synthetic intent recognition only; unsupported/uncertain tasks rejected; not a language generator."}
    (output/"intent_model.json").write_text(json.dumps(model,separators=(",",":")),encoding="utf-8")
    from .text_assistant import IntentModel, handle
    runtime=IntentModel(output/"intent_model.json")
    results=[]
    for text,label in EVALUATION:
        prediction=runtime.predict(text)
        response=handle(text,runtime)
        results.append({"text":text,"expected":label,"predicted":prediction["intent"],
            "probability":prediction["probability"],"status":response["status"]})
    correct=sum(r["expected"]==r["predicted"] for r in results)
    report={"initialization":"random; no pretrained weights","training_examples":len(records),
        "dataset_sha256":hashlib.sha256(Path(data).read_bytes()).hexdigest(),"epochs":epochs,"seed":seed,
        "parameters":int(w.size+b.size),"training_seconds":time.monotonic()-started,
        "fixed_evaluation_accuracy":correct/len(results),"evaluation":results,
        "thresholds":"Conservative fixed defaults, not calibrated probabilities or universal complexity detection.",
        "scope":"Synthetic text intent classifier; no image generation, browsing results, or calendar optimization."}
    (output/"intent_training_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Saved {output}; fixed synthetic evaluation {correct}/{len(results)}",flush=True)
    return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/intent_training.jsonl")
    parser.add_argument("--outdir",default="artifacts_intent")
    parser.add_argument("--epochs",type=int,default=400)
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    if args.epochs<1: parser.error("Epochs must be positive.")
    train(args.data,args.outdir,args.epochs,args.seed)
