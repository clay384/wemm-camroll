"""One-shot, immutable repair attempts and derived comparison records.

The source run and snapshot are read-only. No whole-album build is called.
"""
from __future__ import annotations

import concurrent.futures
import copy
import os
from pathlib import Path
import shutil
import time

from .common import atomic_json, append_jsonl, now_iso, object_hash, read_json, safe_id, sha256
from .dataset import Dataset, snapshot_media
from .metrics import summary
from .runner import Runner


def file_hashes(directory):
    root = Path(directory)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def validate_selection(parent, qa_ids):
    if parent.get('mode') != 'full' or parent.get('status') not in {'completed', 'completed_with_errors'}:
        raise ValueError('repair requires a terminal original full run')
    if not qa_ids or len(set(qa_ids)) != len(qa_ids):
        raise ValueError('repair question selection must be nonempty and unique')
    items = {r['qa_id']: r for r in parent['items']}
    for qa_id in qa_ids:
        if qa_id not in items or items[qa_id].get('execution_status') != 'error':
            raise ValueError('only failed original questions may be repaired')
        if (items[qa_id].get('judge') or {}).get('score') in {0, 1, 2}:
            raise ValueError('valid original scores cannot be replaced')


def verify_core(parent):
    """The adapter may change; paid Agent/Judge behavior must match the frozen release."""
    package = Path(__file__).resolve().parent.parent
    release = package.parents[1]
    frozen = release.parent / safe_id(parent['code_revision']) / 'camroll/camroll-agent'
    if not frozen.is_dir():
        raise ValueError('frozen release unavailable for core verification')
    paths = [str(p.relative_to(package)) for p in (package / 'camroll_agent').rglob('*.py')]
    paths += ['camroll_eval/' + n for n in ('runner.py', 'judge.py', 'judge_prompts.json',
                                            'compat.py', 'task_actions.json', 'dataset.py')]
    for name in paths:
        if sha256(package / name) != sha256(frozen / name):
            raise ValueError('frozen Agent/Judge implementation changed: ' + name)
    return {'frozen_revision': parent['code_revision'], 'matched_files': len(paths)}


def compose(parent, repair):
    """Replace only attempted question records; never splice timing across days."""
    validate_selection(parent, repair['qa_ids'])
    if repair.get('parent_sha256') is None or repair['index_version'] != parent['index_version']:
        raise ValueError('repair provenance does not match parent snapshot')
    if repair.get('status') not in {'completed', 'completed_with_errors'}:
        raise ValueError('only terminal repair attempts can be composed')
    replacements = {r['qa_id']: r for r in repair['items']}
    if set(replacements) != set(repair['qa_ids']):
        raise ValueError('repair must retain a terminal record for every selected question')
    for row in replacements.values():
        if row.get('execution_status') not in {'complete', 'error', 'failed', 'cancelled'}:
            raise ValueError('repair item is not terminal')
        if row.get('mapping_error_count', 0):
            raise ValueError('unknown delivered media IDs; inspect repair before composing')
    result = copy.deepcopy(parent)
    result.update(run_id=repair['run_id'] + '-merged', mode='composite',
        created_at=now_iso(), updated_at=now_iso(), parent_run_id=parent['run_id'],
        repair_run_id=repair['run_id'], parent_sha256=repair['parent_sha256'],
        repair_sha256=repair.get('sealed_sha256'), view_kind='repaired_composite',
        timing_comparable=False, resumed_at=[], repair_qa_ids=repair['qa_ids'],
        repair_status=repair['status'], repair_wall_ms=repair.get('wall_ms'),
        repair_budget_before=repair.get('budget_before'), repair_budget_after=repair.get('budget_after'),
        original_timing_summary={**summary(parent), **copy.deepcopy(parent['summary'])},
        original_run_period={'started_at': parent.get('started_at'), 'finished_at': parent.get('finished_at')},
        note='Quality/retrieval combine unchanged original items with one repair attempt. Timing/resources remain original; repair time/cost are separate.')
    result['items'] = [copy.deepcopy(replacements.get(row['qa_id'], row)) for row in parent['items']]
    result['item_origins'] = {r['qa_id']: repair['run_id'] if r['qa_id'] in replacements else parent['run_id']
                              for r in result['items']}
    result['unchanged_item_hashes'] = {r['qa_id']: object_hash(r) for r in parent['items'] if r['qa_id'] not in replacements}
    result['repaired_valid_count'] = sum(r.get('execution_status') == 'complete' and
        r.get('judge_status') == 'completed' and (r.get('judge') or {}).get('score') in {0, 1, 2}
        for r in replacements.values())
    result['attempted_turn_count_total'] = sum(len(r.get('runtime_turns') or []) for r in parent['items'] + repair['items'])
    result['status'] = 'completed' if result['repaired_valid_count'] == len(replacements) else 'completed_with_errors'
    # Do not leave the original failed per-question status table attached to a composite.
    observed = {r['qa_id']: r for r in result['items']}
    result['planned_status'] = [{**row, 'execution_status': observed[row['qa_id']].get('execution_status'),
        'judge_status': observed[row['qa_id']].get('judge_status'),
        'executed_turn_count': len(observed[row['qa_id']].get('runtime_turns') or [])}
        for row in parent.get('qa_plan', [])]
    result['summary'] = summary(result)
    return result


class RepairRunner(Runner):
    def run_repair(self, parent_id, qa_ids, *, run_id, verify_frozen=True):
        self.acquire()
        parent_path = self.root / 'runs' / safe_id(parent_id) / 'run.json'
        writable = False
        backend = None
        sampler = None
        started = time.monotonic()
        try:
            parent = read_json(parent_path)
            parent_hash = sha256(parent_path)
            validate_selection(parent, qa_ids)
            dataset = Dataset(self.source)
            integrity = dataset.integrity()
            for key in ('dataset_sha256', 'qa_sha256', 'task_action_policy_sha256'):
                if integrity[key] != parent['input_integrity'][key]:
                    raise ValueError('repair input changed: ' + key)
            core = verify_core(parent) if verify_frozen else {'test_only': True}
            judge = self.judge_factory(cancel=self.cancel, on_call=self._judge_call)
            if judge.protocol() != parent['judge_protocol']:
                raise ValueError('repair Judge protocol changed')
            version = safe_id(parent['index_version'])
            original_memory = self.albums / safe_id(parent['album_id'])
            snapshot = original_memory / 'snapshots' / version
            frozen_hashes = file_hashes(snapshot)
            if (original_memory / 'CURRENT').read_text().strip() != version:
                raise ValueError('original snapshot pointer changed')
            rows = [row for row in dataset.rows if row['qa_id'] in qa_ids]
            if len(rows) != len(qa_ids):
                raise ValueError('repair questions missing from frozen dataset')
            self._path = self.root / 'runs' / safe_id(run_id)
            if self._path.exists():
                raise ValueError('repair run already exists; no paid retry allowed')
            claims = self.root / 'repair_claims' / safe_id(parent_id)
            claims.mkdir(parents=True, exist_ok=True)
            if any((claims / (safe_id(q) + '.json')).exists() for q in qa_ids):
                raise ValueError('a selected question already has a repair attempt')
            self._path.mkdir(mode=0o700)
            # Claims survive errors/interruption. An uncertain paid call must not be retried implicitly.
            for q in qa_ids:
                with (claims / (safe_id(q) + '.json')).open('x') as f:
                    import json
                    json.dump({'run_id': run_id, 'created_at': now_iso()}, f)
                (claims / (q + '.json')).chmod(0o600)
            memory = self._path / 'pinned_memory'
            shutil.copytree(snapshot, memory / 'snapshots' / version)
            (memory / 'CURRENT').write_text(version + '\n')
            media, metadata = snapshot_media(memory, version)
            self._state = {'run_id': run_id, 'system_id': 'camroll_wemm', 'mode': 'repair',
                'view_kind': 'repair_attempt', 'formal_protocol': False, 'status': 'preparing',
                'created_at': now_iso(), 'started_at': now_iso(), 'parent_run_id': parent_id,
                'parent_sha256': parent_hash, 'index_version': version, 'snapshot_hashes': frozen_hashes,
                'core_verification': core, 'code_revision': os.getenv('CAMROLL_RELEASE_REVISION', 'unrecorded'),
                'config': parent['config'], 'qa_concurrency': parent['qa_concurrency'],
                'judge_concurrency': parent['judge_concurrency'], 'qa_ids': qa_ids, 'qa_count': len(rows),
                'qa_plan': [{'qa_id': r['qa_id'], 'question': r['question'], 'planned_turn_count': len(dataset.turns(r))} for r in rows],
                'planned_turn_count': sum(len(dataset.turns(r)) for r in rows), 'items': [],
                'input_integrity': integrity, 'judge_protocol': parent['judge_protocol'],
                'judge_model': parent['judge_model'], 'media_mapping': media, 'timing_comparable': False,
                'metric_source': parent['metric_source'], 'phases': {'qa_eval': {}}, 'cache_mode': 'original_snapshot'}
            writable = True
            self._save()
            atomic_json(self._path / 'questions.json', rows)
            backend = self.backend_factory()
            self._state['budget_before'] = backend.budget()
            if self.sample_resources:
                from .telemetry import ResourceSampler
                sampler = ResourceSampler(self._path / 'resources.jsonl', lambda: self._state['status'])
                sampler.start()
            phase = self._state['phases']['qa_eval']
            phase.update(started_at=now_iso(), agent_phase_started_at=now_iso(), agent_phase_started_at_epoch=time.time())
            self._state['status'] = 'evaluating'
            self._save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=parent['judge_concurrency']) as judges:
                with concurrent.futures.ThreadPoolExecutor(max_workers=parent['qa_concurrency']) as agents:
                    pending = {agents.submit(self._evaluate, row, backend, memory, version,
                        parent['config'], media, run_id): row for row in rows}
                    judging = []
                    for future in concurrent.futures.as_completed(pending):
                        row = pending[future]
                        item = future.result()
                        item['repair_provenance'] = {'parent_run_id': parent_id, 'repair_run_id': run_id,
                            'original_item_sha256': object_hash(next(r for r in parent['items'] if r['qa_id'] == row['qa_id'])),
                            'attempt': 1, 'snapshot_version': version}
                        self._item(item)
                        judging.append(judges.submit(self._score, copy.deepcopy(item), row, judge))
                phase.update(agent_phase_finished_at=now_iso(), agent_phase_finished_at_epoch=time.time())
                phase['agent_phase_wall_ms'] = (phase['agent_phase_finished_at_epoch'] - phase['agent_phase_started_at_epoch']) * 1000
                self._state['status'] = 'judging'
                self._save()
                for future in judging:
                    future.result()
            if sha256(parent_path) != parent_hash or file_hashes(snapshot) != frozen_hashes:
                raise ValueError('original run or snapshot changed during repair')
            self._state['source_unchanged'] = True
            self._state['status'] = 'completed' if all(r.get('execution_status') == 'complete' and
                r.get('judge_status') == 'completed' for r in self._state['items']) else 'completed_with_errors'
            phase.update(status='done', finished_at=now_iso())
        except Exception as exc:
            if writable:
                self._state.update(status='failed', fatal_error={'type': type(exc).__name__})
            raise
        finally:
            if sampler:
                sampler.stop()
                self._state['resource_summary'] = sampler.summary()
            if writable:
                if backend:
                    self._state['budget_after'] = backend.budget()
                self._state.update(finished_at=now_iso(), wall_ms=(time.monotonic() - started) * 1000)
                self._save()
            self.release()
        sealed = copy.deepcopy(self._state)
        sealed['sealed_sha256'] = sha256(self._path / 'run.json')
        merged = compose(parent, sealed)
        target = self.root / 'runs' / merged['run_id']
        target.mkdir(mode=0o700)
        atomic_json(target / 'run.json', merged)
        atomic_json(target / 'verification.json', {'parent_sha256': parent_hash,
            'repair_sha256': sealed['sealed_sha256'], 'unchanged_items': merged['unchanged_item_hashes'],
            'original_snapshot_hashes': frozen_hashes, 'index_version': version})
        return merged


def validate_photo(row):
    """asset_id identifies the original, while row.path can be an EXIF/RGB view."""
    import hashlib
    import io
    from PIL import Image, ImageOps
    view = Path(row['path'])
    view_hash = sha256(view)
    if view_hash == row['asset_id']:
        return {'source_sha256': view_hash, 'view_sha256': view_hash, 'view_is_original': True}
    sources = row.get('sources') or []
    source = Path(sources[0]['source_path']) if sources else view.parent / 'original.jpg'
    if sha256(source) != row['asset_id']:
        raise ValueError('original photo bytes changed')
    # Verify the existing derived view in memory; never rewrite preprocessing outputs.
    with Image.open(source) as image:
        corrected = ImageOps.exif_transpose(image).convert('RGB')
        buf = io.BytesIO()
        corrected.save(buf, format='JPEG', quality=95, subsampling=0)
    if hashlib.sha256(buf.getvalue()).hexdigest() != view_hash:
        raise ValueError('derived photo differs from frozen preprocessing')
    return {'source_sha256': row['asset_id'], 'view_sha256': view_hash, 'derivation_verified': True}


def repair_caption(root, albums, parent_id, *, repair_id, resume_preflight=False):
    """Only the failed caption and its text vector; original events/vectors untouched."""
    import contextlib
    import numpy as np
    from camroll_agent.pipeline import _caption, _vector, _publish, _manifest
    from .runner import CamrollBackend
    guard = Runner(root, '.', albums)
    guard.acquire()
    out = None
    backend = None
    began = time.monotonic()
    try:
        parent_path = Path(root) / 'runs' / safe_id(parent_id) / 'run.json'
        parent = read_json(parent_path)
        core = verify_core(parent)
        original = Path(albums) / safe_id(parent['album_id'])
        version = safe_id(parent['index_version'])
        snapshot = original / 'snapshots' / version
        hashes, parent_hash = file_hashes(snapshot), sha256(parent_path)
        data = read_json(snapshot / 'records.json')
        errors = [e for e in parent['build_summary'].get('errors', []) if e['stage'] == 'caption']
        if len(errors) != 1:
            raise ValueError('expected exactly one original caption failure')
        row = next(r for r in data['images'] if r['id'] == errors[0]['image_id'])
        if row.get('caption'):
            raise ValueError('original image already has a caption')
        path = Path(root) / 'caption_repairs' / safe_id(parent_id)
        if path.exists():
            previous = read_json(path / 'repair.json')
            if not (resume_preflight and previous.get('status') == 'failed' and
                    not previous.get('paid_call_started', False) and
                    previous.get('budget_before') == previous.get('budget_after') and
                    not (path / 'calls.jsonl').exists() and not (path / 'caption.json').exists()):
                raise FileExistsError('caption attempt exists; paid or uncertain calls cannot be retried')
            atomic_json(path / ('preflight-failure-' + object_hash(previous)[:16] + '.json'), previous)
        else:
            path.mkdir(parents=True, mode=0o700)
        out = {'repair_id': safe_id(repair_id), 'parent_run_id': parent_id, 'parent_sha256': parent_hash,
            'original_index_version': version, 'image_id': row['id'], 'status': 'running',
            'created_at': now_iso(), 'core_verification': core, 'event_membership_repaired': False,
            'paid_call_started': False,
            'note': 'Only caption/text embedding repaired; existing event membership and event vectors preserved.'}
        atomic_json(path / 'repair.json', out)
        backend = CamrollBackend()
        out['budget_before'] = backend.budget()
        if backend.embedding.fingerprint != data['metadata']['embedding_fingerprint']:
            raise ValueError('embedding fingerprint changed')
        out['photo_verification'] = validate_photo(row)
        _, profile, _ = _manifest(parent_path.parent / 'build_spec.json')
        def capture(call):
            append_jsonl(path / 'calls.jsonl', call)
        with contextlib.ExitStack() as stack:
            stack.enter_context(backend.qwen.capture_calls(capture))
            stack.enter_context(backend.embedding.capture_calls(capture))
            out['paid_call_started'] = True
            atomic_json(path / 'repair.json', out)
            value = _caption(backend.qwen, row, profile)
            atomic_json(path / 'caption.json', value)  # retained even if embedding subsequently fails
            text = (value['caption'] + '\n' + value['visible_text']).strip()
            chunks = backend.embedding.split_text(text)
            if not chunks:
                raise ValueError('empty caption chunks')
            additions = [_vector(backend.embedding, text=chunk) for chunk in chunks]
        row.update(caption=value['caption'], visible_text=value['visible_text'], caption_status='ok',
                   observed_location=value['location'], observed_people=value['people'])
        with np.load(snapshot / 'vectors.npz', allow_pickle=False) as z:
            vectors = {k: z[k].tolist() for k in z.files}
        assert row['id'] not in data['vector_ids']['caption']
        vectors['caption'].extend(additions)
        data['vector_ids']['caption'].extend([row['id']] * len(additions))
        meta = data['metadata']
        meta.update(caption_count=meta['caption_count'] + 1, text_embedding_count=meta['text_embedding_count'] + 1,
            text_coverage=(meta['text_embedding_count'] + 1) / meta['image_count'],
            caption_repair={'parent_run_id': parent_id, 'repair_id': repair_id, 'original_snapshot': version,
                            'event_membership_repaired': False})
        # Original failure facts remain available in metadata; the repair is explicit.
        derived = path / 'derived_memory'
        new_version = _publish(derived, data['images'], data['events'], data['vector_ids'], vectors, meta)
        if file_hashes(snapshot) != hashes or sha256(parent_path) != parent_hash:
            raise ValueError('original changed during caption repair')
        published = read_json(derived / 'snapshots' / new_version / 'records.json')
        previous = read_json(snapshot / 'records.json')
        assert object_hash(published['events']) == object_hash(previous['events'])
        assert all(object_hash(a) == object_hash(b) for a, b in zip(previous['images'], published['images']) if a['id'] != row['id'])
        for branch in ('image', 'event'):
            assert published['metadata']['vector_hashes'][branch] == previous['metadata']['vector_hashes'][branch]
        out.update(status='completed', derived_index_version=new_version, source_unchanged=True,
            untouched_images=len(data['images']) - 1, original_snapshot_hashes=hashes,
            caption_count=meta['caption_count'], text_embedding_count=meta['text_embedding_count'])
    except Exception as exc:
        if out:
            out.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        if out:
            if backend:
                out['budget_after'] = backend.budget()
            out.update(finished_at=now_iso(), wall_ms=(time.monotonic() - began) * 1000)
            atomic_json(path / 'repair.json', out)
        guard.release()
    return out
