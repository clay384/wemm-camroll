"""Official WeMM Transformers encoding with explicit bounded preprocessing."""
import os,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import yaml
CONFIG=yaml.safe_load((ROOT/'config.yaml').read_text())
os.environ['CUDA_VISIBLE_DEVICES']=CONFIG['gpu_uuid']
os.environ['HF_HOME']=str(ROOT/'.hf')
os.environ['TRITON_CACHE_DIR']=str(ROOT/'.triton')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'.torchinductor')
os.environ['TORCH_HOME']=str(ROOT/'.torch')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']=str(CONFIG['cpu_threads'])
import torch
import numpy as np
from transformers import AutoModel,AutoProcessor
from qwen_vl_utils import process_vision_info

def fingerprint():
    p=ROOT/CONFIG['model_path']
    parts={k:CONFIG[k] for k in ['model_id','revision','precision','dimension','profiles','profile','video_segment_seconds']}
    parts['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parts['pipeline_sha256']={f:hashlib.sha256((ROOT/'backend'/f).read_bytes()).hexdigest() for f in ['preprocess.py','worker.py']}
    parts['candidate_modes']=['image','visual_pdf','native_video','text_chunk','image_text_joint']
    parts['files']={f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['config.json','modeling_wemm_embedding.py','tokenizer.json','tokenizer_config.json','processor_config.json','chat_template.jinja','embedding_chat_template.jinja']}
    parts.update(normalize='float32 L2',aggregation='max unit score; ties asset ID',split='pack whole paragraphs to 512 tokens; oversized paragraph fixed token windows, no overlap',weights='b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4')
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode()).hexdigest(),parts

class Encoder:
 def __init__(self):
    torch.set_num_threads(CONFIG['cpu_threads'])
    self.status='loading'; t=time.perf_counter()
    p=str(ROOT/CONFIG['model_path'])
    self.processor=AutoProcessor.from_pretrained(p,trust_remote_code=True,local_files_only=True)
    self.processor.tokenizer.padding_side='right'
    # Reserve memory for existing users. Never allocate the second GPU.
    free,total=torch.cuda.mem_get_info()
    allowed=free-CONFIG['gpu_reserve_mib']*1024**2
    if allowed<18*1024**3: raise RuntimeError('BLOCKED: shared GPU free memory below BF16 safety budget')
    torch.cuda.set_per_process_memory_fraction(min(allowed/total,0.86))
    self.model=AutoModel.from_pretrained(p,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    self.model.config.use_cache=False
    torch.cuda.synchronize(); self.load_seconds=time.perf_counter()-t
    self.load_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20
    self.load_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20
    self.status='ready'; self.calls=0
    self.fingerprint,self.spec=fingerprint()
 def messages(self,text=None,image=None,video=None,profile=None):
    b=CONFIG['profiles'][profile or CONFIG['profile']]; c=[]
    if image: c.append({'type':'image','image':str(image),'min_pixels':65536,'max_pixels':b['image_pixels']})
    if video: c.append({'type':'video','video':str(video),'min_pixels':4096,'max_pixels':b['video_pixels'],'nframes':b['video_frames']})
    if text: c.append({'type':'text','text':text})
    if not c: raise ValueError('输入为空')
    return [{'role':'user','content':c}]
 def encode(self,text=None,image=None,video=None,profile=None):
    t=time.perf_counter(); self.status='busy'
    try:
     if text and len(self.processor.tokenizer.encode(text,add_special_tokens=False))>CONFIG['profiles'][profile or CONFIG['profile']]['text_tokens']:
      raise ValueError('文本超过 512 token；候选需先分块，查询请缩短')
     messages=self.messages(text,image,video,profile)
     prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
     images,videos,kw=process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
     if videos is not None: videos,metadata=map(list,zip(*videos))
     else: metadata=None
     inp=self.processor(text=prompt,images=images,videos=videos,video_metadata=metadata,return_tensors='pt',**kw)
     last=int(inp.input_ids[0,-1]); token=self.processor.tokenizer.convert_ids_to_tokens(last)
     if token!='<embedding>': raise AssertionError(f'Wrong final token: {token!r}')
     inp=inp.to('cuda:0'); torch.cuda.synchronize(); pre=time.perf_counter()-t
     torch.cuda.reset_peak_memory_stats()
     t1=time.perf_counter()
     with torch.inference_mode(): out=self.model.embedding(**inp,use_cache=False).float()
     out=torch.nn.functional.normalize(out,dim=-1)
     torch.cuda.synchronize(); elapsed=time.perf_counter()-t1
     vec=out[0].cpu().numpy()
     if vec.shape!=(4096,) or not np.isfinite(vec).all(): raise AssertionError('Invalid embedding')
     self.calls+=1
     stats={'preprocess_s':pre,'embedding_s':elapsed,'total_s':time.perf_counter()-t,'tokens':int(inp.input_ids.shape[1]),'last_token':token,'input_shapes':{k:list(v.shape) for k,v in inp.items() if hasattr(v,'shape')},'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'fingerprint':self.fingerprint,'cache_hit':False,'profile':profile or CONFIG['profile']}
     stats['video_metadata']=json.loads(json.dumps(metadata,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
     stats['image_grid_thw']=inp['image_grid_thw'].cpu().tolist() if 'image_grid_thw' in inp else None
     stats['video_grid_thw']=inp['video_grid_thw'].cpu().tolist() if 'video_grid_thw' in inp else None
     return vec,stats
    finally: self.status='ready'

 def reference_encode(self,text=None,image=None,video=None,profile=None):
    import importlib.util
    spec=importlib.util.spec_from_file_location('wemm_official_reference',ROOT/'official-locked/examples/transformers_inference.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    self.status='busy';t=time.perf_counter()
    try:
     out=mod.encode(self.processor,self.model,self.messages(text,image,video,profile),'cuda:0',4096)
     torch.cuda.synchronize();self.calls+=1
     return out.numpy()[0],{'reference':True,'total_s':time.perf_counter()-t,'fingerprint':self.fingerprint}
    finally:self.status='ready'
