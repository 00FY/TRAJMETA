"""Package or restore exact experiment evidence without refitting models."""
import argparse
import gzip
import json
from pathlib import Path
from hashlib import sha256
import zipfile

def digest(data): return sha256(data).hexdigest()

def install(path,data):
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f'Refusing to overwrite changed evidence: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['pack','restore']); a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; out=root/'reports/study-v1'
    archive=out/'experiment_evidence.zip'; manifest=out/'evidence_manifest.json'
    if a.command=='pack':
        if archive.exists() or manifest.exists(): raise FileExistsError('Published evidence already exists')
        files=sorted(p for p in (root/'results/study-v1').rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl'})
        hashes={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in files}
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in files: z.write(path,path.relative_to(root).as_posix())
        manifest.write_text(json.dumps({'archive_sha256':digest(archive.read_bytes()),'files':hashes,'excluded':'Per-run PPO policy ZIP files; not needed to verify recorded results.'},indent=2))
        print(f'Packed {len(files)} evidence files ({archive.stat().st_size} bytes).')
    else:
        expected=json.loads(manifest.read_text())
        assert digest(archive.read_bytes())==expected['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert set(z.namelist())==set(expected['files'])
            for name,h in expected['files'].items():
                destination=(root/name).resolve()
                if not destination.is_relative_to(root/'results/study-v1'): raise ValueError('Invalid archive path')
                data=z.read(name); assert digest(data)==h; install(destination,data)
        models=root/'models/study-v1'; versions=json.loads((models/'versions.json').read_text())
        bank=gzip.decompress((models/'bank.jsonl.gz').read_bytes()); assert digest(bank)==versions['bank_sha256']
        encoder=(models/'encoder.pt').read_bytes(); assert digest(encoder)==versions['encoder_sha256']
        artifacts=root/'artifacts/study-v1'
        install(artifacts/'bank.jsonl',bank); install(artifacts/'encoder/encoder.pt',encoder)
        install(artifacts/'encoder/sanity.json',(out/'encoder_sanity.json').read_bytes())
        install(artifacts/'encoder/training.json',(out/'encoder_training.json').read_bytes())
        print(f"Verified/restored {len(expected['files'])} evidence files and frozen artifacts.")

if __name__=='__main__': main()
