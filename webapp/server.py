"""Local-only TrajMeta workspace. Launch with the project's virtual environment."""
import argparse
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, parse_qs
import uuid

ROOT=Path(__file__).resolve().parents[1]
STATIC=Path(__file__).parent/'static'
REPORTS=ROOT/'reports/study-v1'
METHODS=['random','optuna','ppo','metadata','retrieval','shaped','final_config','simple_retrieval','cold']
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,allow_nan=False),encoding='utf-8'); temp.replace(path)

class Jobs:
    def __init__(self, root=ROOT):
        self.root=root; self.directory=root/'results/ui'; self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock(); self.processes={}
        self.datasets=read(root/'data/audits/study-v1/audit.json')
        self.allowed={'synthetic_val'}|{r['dataset_key'] for r in self.datasets if r['split']=='meta_train'}
        for path in self.directory.glob('*/job.json'):
            job=read(path)
            if job['status']=='running':
                job.update(status='interrupted',error='The server stopped before completion. Start a new run to retry.'); save(path,job)

    def start(self,request):
        if not isinstance(request,dict): raise ValueError('Expected a JSON object.')
        method=request.get('method','shaped'); dataset=request.get('dataset','synthetic_val')
        budget=request.get('budget',8); seed=request.get('seed',2026)
        if not isinstance(method,str) or method not in METHODS: raise ValueError('Unknown search method.')
        if not isinstance(dataset,str) or dataset not in self.allowed|{'upload'}: raise ValueError('Choose a development dataset. Held-out tasks are sealed.')
        if type(budget) is not int or not 1<=budget<=32: raise ValueError('Budget must be an integer from 1 to 32.')
        if type(seed) is not int or not 0<=seed<2**31: raise ValueError('Seed must be an integer from 0 to 2147483647.')
        if dataset=='upload' and (not isinstance(request.get('csv'),str) or not isinstance(request.get('target'),str) or not request.get('target')):
            raise ValueError('Upload a CSV and choose its target column.')
        with self.lock:
            for key in list(self.processes): self.refresh(key)
            if any(p.poll() is None for p,_ in self.processes.values()): raise RuntimeError('A search is already running. Finish or cancel it first.')
            key=uuid.uuid4().hex; folder=self.directory/key; folder.mkdir()
            job={'id':key,'method':method,'dataset':dataset,'budget':budget,'seed':seed,'status':'running','created_at':datetime.now(timezone.utc).isoformat()}
            save(folder/'request.json',{**request,'method':method,'dataset':dataset,'budget':budget,'seed':seed})
            save(folder/'job.json',job)
            try:
                with (folder/'worker.log').open('w',encoding='utf-8') as log:
                    env=os.environ.copy(); env['PYTHONPATH']=str(self.root/'src'); env['OMP_NUM_THREADS']='1'
                    process=subprocess.Popen([sys.executable,str(self.root/'webapp/worker.py'),str(folder)],cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
                self.processes[key]=(process,time.monotonic())
            except Exception as exc:
                job.update(status='failed',error=str(exc)); save(folder/'job.json',job)
                raise
            return job

    def folder(self,key):
        if not re.fullmatch('[a-f0-9]{32}',key): raise ValueError('Invalid run ID.')
        folder=self.directory/key
        if not (folder/'job.json').exists(): raise FileNotFoundError('Run not found.')
        return folder

    def stop_process(self,key):
        pair=self.processes.get(key)
        if pair and pair[0].poll() is None:
            if os.name=='nt': subprocess.run(['taskkill','/PID',str(pair[0].pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            else: os.killpg(pair[0].pid,signal.SIGTERM)
            pair[0].wait(timeout=15)

    def refresh(self,key):
        folder=self.folder(key); job=read(folder/'job.json'); pair=self.processes.get(key)
        if job['status']=='running' and pair:
            process,start=pair
            if process.poll() is None and time.monotonic()-start>900:
                self.stop_process(key); job.update(status='failed',error='The search exceeded the 15-minute app limit.')
            elif process.poll() is not None:
                if process.returncode==0 and (folder/'result.json').exists(): job['status']='completed'
                else:
                    error=read(folder/'error.json')['error'] if (folder/'error.json').exists() else 'The worker stopped unexpectedly. See results/ui for its log.'
                    job.update(status='failed',error=error)
            save(folder/'job.json',job)
        return job

    def detail(self,key):
        with self.lock:
            job=self.refresh(key); folder=self.folder(key); records=[]
            if (folder/'evaluations.jsonl').exists():
                for line in (folder/'evaluations.jsonl').read_text(encoding='utf-8').splitlines():
                    try: records.append(json.loads(line))
                    except json.JSONDecodeError: pass
            job['records']=records
            if (folder/'result.json').exists() and job['status']=='completed': job['result']=read(folder/'result.json')
            return job

    def cancel(self,key):
        with self.lock:
            job=self.refresh(key)
            if job['status']=='running':
                self.stop_process(key); job.update(status='cancelled'); save(self.folder(key)/'job.json',job)
            return job

    def history(self):
        with self.lock:
            paths=sorted(self.directory.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]
            return [self.refresh(p.parent.name) for p in paths]

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def valid_host(self): return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
    def reply(self,value,status=200):
        self.send_bytes(json.dumps(value,allow_nan=False).encode(),'application/json',status)
    def send_bytes(self,body,kind,status=200,download=None):
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if download: self.send_header('Content-Disposition',f'attachment; filename="{download}"')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if not self.valid_host(): return self.reply({'error':'Invalid local host.'},403)
        try:
            url=urlparse(self.path); path=url.path; query=parse_qs(url.query)
            if path=='/api/overview':
                runs=read(REPORTS/'test_results.json')
                return self.reply({'statistics':read(REPORTS/'statistics.json'),'verification':read(REPORTS/'verification.json'),'encoder':read(REPORTS/'encoder_sanity.json'),'protocol':read(REPORTS/'final_protocol.json'),'datasets':self.server.jobs.datasets,'methods':METHODS,'runs':[{k:v for k,v in r.items() if k!='records'} for r in runs]})
            if path=='/api/bank':
                dataset=query.get('dataset',[''])[0]; page=max(0,int(query.get('page',['0'])[0])); key=query.get('id',[''])[0]
                bank=self.server.bank
                if key:
                    found=next((r for r in bank if r['trajectory_id']==key),None)
                    return self.reply(found if found else {'error':'Trajectory not found.'},200 if found else 404)
                rows=[r for r in bank if not dataset or r['dataset_key']==dataset]
                rows=sorted(rows,key=lambda r:-r['inner_score'])
                return self.reply({'total':len(rows),'page':page,'items':[{k:r[k] for k in ['trajectory_id','dataset_key','seed','inner_score','pipeline_config']} for r in rows[page*12:page*12+12]]})
            if path=='/api/jobs': return self.reply(self.server.jobs.history())
            if path.startswith('/api/jobs/'): return self.reply(self.server.jobs.detail(path.rsplit('/',1)[1]))
            if path.startswith('/download/'):
                files={'report':'REPORT.md','summary':'method_summary.csv','runs':'test_runs.csv','statistics':'statistics.json'}
                name=files.get(path.rsplit('/',1)[1])
                if not name: raise FileNotFoundError('Download not found.')
                return self.send_bytes((REPORTS/name).read_bytes(),'application/octet-stream',download=name)
            assets={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in assets: raise FileNotFoundError('Page not found.')
            file=STATIC/assets[path]; self.send_bytes(file.read_bytes(),mimetypes.guess_type(str(file))[0] or 'text/plain')
        except (ValueError,FileNotFoundError) as exc: self.reply({'error':str(exc)},404 if isinstance(exc,FileNotFoundError) else 400)
        except Exception: self.reply({'error':'Unable to load project evidence. Check that the repository files are present.'},500)
    def do_POST(self):
        origin=self.headers.get('Origin')
        if not self.valid_host() or (origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}): return self.reply({'error':'Only this local app may submit runs.'},403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=2_000_000: raise ValueError('Request must be smaller than 2 MB.')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json': raise ValueError('Expected application/json.')
            request=json.loads(self.rfile.read(length)); path=urlparse(self.path).path
            if path=='/api/jobs': return self.reply(self.server.jobs.start(request),201)
            if path.startswith('/api/jobs/') and path.endswith('/cancel'): return self.reply(self.server.jobs.cancel(path.split('/')[-2]))
            raise FileNotFoundError('Endpoint not found.')
        except RuntimeError as exc: self.reply({'error':str(exc)},409)
        except (ValueError,FileNotFoundError) as exc: self.reply({'error':str(exc)},400)
        except Exception: self.reply({'error':'Could not start the worker. Check the project virtual environment.'},500)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8765); args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.jobs=Jobs(); server.bank=[json.loads(line) for line in gzip.decompress((ROOT/'models/study-v1/bank.jsonl.gz').read_bytes()).decode().splitlines()]
    print(f'TrajMeta workspace: http://127.0.0.1:{server.server_port}',flush=True)
    try: server.serve_forever()
    finally:
        for key in list(server.jobs.processes): server.jobs.cancel(key)
        server.server_close()

if __name__=='__main__': main()
