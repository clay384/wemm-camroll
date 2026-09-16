"""Small billed capability probe, all calls counted in the shared ledger."""
import json
import sys
from pathlib import Path

from PIL import Image
from camroll_agent.qwen import QwenClient, image_messages

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
picture=out/'probe-red.jpg'
Image.new('RGB',(64,64),'red').save(picture)
client=QwenClient()
schema={'type':'object','properties':{'color':{'type':'string'}},'required':['color'],'additionalProperties':False}
results={}
with client.request_scope(max_requests=12,deadline_s=180) as scope:
    results['text']=client.chat([{'role':'user','content':'只回复：文本调用成功'}],role='probe')['content']
    results['vision_schema']=client.generate_json('这张图片的颜色是什么？用JSON的color字段回答。',[str(picture)],schema,role='probe')
    assert any(x in results['vision_schema']['color'].lower() for x in ['red','红'])
    tool={'type':'function','function':{'name':'lookup','description':'查找照片','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}}
    answer=client.chat([{'role':'user','content':'请调用lookup搜索红色汽车。'}],tools=[tool],tool_choice={'type':'function','function':{'name':'lookup'}},role='probe')
    calls=answer.get('tool_calls') or []
    assert calls and calls[0]['function']['name']=='lookup'
    results['tool_call']={'name':calls[0]['function']['name'],'arguments':json.loads(calls[0]['function']['arguments'])}
    clean={k:answer[k] for k in ['role','content','tool_calls'] if k in answer}
    roundtrip=client.chat([{'role':'user','content':'请调用lookup搜索红色汽车。'},clean,
                         {'role':'tool','tool_call_id':calls[0]['id'],'content':'{"results":[{"id":"img_probe","caption":"红色汽车"}]}'}],role='probe')
    assert roundtrip.get('content')
    results['tool_roundtrip']=roundtrip['content']
    results['stream']=client.chat([{'role':'user','content':'只回复：流式成功'}],role='probe',stream=True)['content']
    results['scope']=scope.summary()
results['capabilities']=client.capabilities()
results['budget']=client.usage_summary()
(out/'qwen_capabilities.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False))
