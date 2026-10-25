"""Local API boundary tests; no model training or study mutation."""
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location('webapp_server', Path(__file__).resolve().parents[1] / 'webapp/server.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)

@pytest.fixture
def service(tmp_path):
    audit = tmp_path / 'data/audits/study-v1/audit.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps([{'dataset_key': 'train', 'split': 'meta_train'}, {'dataset_key': 'test', 'split': 'meta_test'}]))
    server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
    server.jobs = web.Jobs(tmp_path)
    server.bank = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()

def request(service, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(f'http://127.0.0.1:{service.server_port}{path}', data=data, headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read(), response.headers

@pytest.mark.parametrize('body', [[], {'dataset':[]}, {'method':{}}, {'dataset':'test'}, {'budget':True}, {'budget':33}, {'budget':0}, {'seed':-1}, {'seed':1.2}, {'method':'unknown'}, {'dataset':'upload'}])
def test_invalid_jobs_do_not_create_workers(service, body):
    status, _, _ = request(service, '/api/jobs', body)
    assert status == 400
    assert not service.jobs.processes
    assert list(service.jobs.directory.iterdir()) == []

def test_cross_origin_and_rebinding_rejected(service):
    assert request(service, '/api/jobs', {}, {'Origin':'https://example.com'})[0] == 403
    assert request(service, '/api/jobs', headers={'Host':'example.com'})[0] == 403

def test_no_arbitrary_file_access(service):
    for path in ['/../README.md', '/download/README.md', '/.env']:
        assert request(service, path)[0] == 404
    assert request(service, '/api/jobs/not-a-run')[0] == 400

def test_assets_and_empty_history(service):
    status, body, headers = request(service, '/')
    assert status == 200 and b'TrajMeta' in body
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    assert json.loads(request(service, '/api/jobs')[1]) == []

def test_restart_marks_incomplete_run_and_reads_partial_log(service):
    key = 'a' * 32
    folder = service.jobs.directory / key
    folder.mkdir()
    web.save(folder / 'job.json', {'id':key,'status':'running'})
    (folder / 'evaluations.jsonl').write_text('{"evaluation":1}\n{"evaluation":')
    restored = web.Jobs(service.jobs.root)
    detail = restored.detail(key)
    assert detail['status'] == 'interrupted'
    assert detail['records'] == [{'evaluation':1}]
