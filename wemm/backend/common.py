from pathlib import Path
import yaml,sqlite3,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]; CONFIG=yaml.safe_load((ROOT/'config.yaml').read_text())
for name in ['data/originals','data/derived','data/indexes','data/tmp','reports','logs','run','import','benchmark']:(ROOT/name).mkdir(parents=True,exist_ok=True)
def db():
 c=sqlite3.connect(ROOT/'data/lab.sqlite',timeout=30);c.row_factory=sqlite3.Row;c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA foreign_keys=ON');return c
def init_db():
 with db() as c:c.executescript('''
 CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT NOT NULL,paused INTEGER DEFAULT 0);
 INSERT OR IGNORE INTO datasets VALUES('demo','公开演示',0);
 INSERT OR IGNORE INTO datasets VALUES('personal','我的导入',0);
 INSERT OR IGNORE INTO datasets VALUES('queries','查询图片',0);
 CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,ext TEXT,bytes INTEGER,status TEXT,error TEXT,created REAL,preprocess_s REAL DEFAULT 0);
 CREATE TABLE IF NOT EXISTS members(dataset TEXT,asset TEXT,name TEXT,PRIMARY KEY(dataset,asset,name));
 CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,asset TEXT,kind TEXT,path TEXT,thumb TEXT,meta TEXT,vector TEXT,fingerprint TEXT,stats TEXT);
 CREATE TABLE IF NOT EXISTS annotations(query TEXT,unit TEXT,label TEXT,created REAL);
 CREATE TABLE IF NOT EXISTS events(time REAL,kind TEXT,payload TEXT);
 CREATE INDEX IF NOT EXISTS idx_units_asset ON units(asset);
 CREATE INDEX IF NOT EXISTS idx_members_asset ON members(asset);
 PRAGMA optimize;
 ''')
def event(kind,payload):
 with db() as c:c.execute('INSERT INTO events VALUES(?,?,?)',(time.time(),kind,json.dumps(payload,ensure_ascii=False)))
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def relative_name(name):
 from pathlib import PurePosixPath
 if not name or len(name)>1000 or '\x00' in name or '\\' in name:raise ValueError('无效相对路径')
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or ':' in p.parts[0]:raise ValueError('禁止路径穿越或绝对路径')
 return str(p)
