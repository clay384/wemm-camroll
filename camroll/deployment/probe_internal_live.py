import json,sqlite3,time
from pathlib import Path
import numpy as np
from camroll_agent.wemm import EmbeddingClient,WeMMError
root=Path('/home/realmagic/camroll-wemm')
client=EmbeddingClient()
for _ in range(12):
    try:
        fingerprint=client.fingerprint
        break
    except WeMMError:
        time.sleep(2)
else:
    raise SystemExit('worker not ready')
text='照片中的红色汽车。'
a=client.encode(text=text)
photo=next((root/'runtime/albums/photobench-album3/assets').glob('*/*.wemm.jpg'),None)
if photo is None:
    photo=next((root/'runtime/albums/photobench-album3/assets').glob('*/wemm*.jpg'))
b=client.encode(image_path=photo)
longtext='这是无损中文分块测试，包含照片与事件。'*80
chunks=client.split_text(longtext)
assert ''.join(chunks)==longtext and len(chunks)>1
try:
    client.encode(text=longtext)
except WeMMError as exc:
    assert exc.code=='query_too_long'
else:
    raise AssertionError('long query not rejected')
report={'fingerprint':fingerprint,'text_dimension':len(a),'image_dimension':len(b),'text_norm':float(np.linalg.norm(a)),'image_norm':float(np.linalg.norm(b)),'split_count':len(chunks),'lossless':True,'long_query_rejected':True,'image_asset':photo.parent.name,'timings':client.last_timings}
(root/'reports/20260915-221637/wemm_api_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
