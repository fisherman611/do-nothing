"""Run one Modal CLI process per model to isolate Windows client connections."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from asr.registry import MODELS
from metrics.io import checked_path, load_json, write_json

# Aggregate caps: 7.65 USD + 0.4 repeat checks. Prior attempts are recorded separately.
CAPS = {"phowhisper-medium": 1.60, "phowhisper-large": 1.85,
        "whisper-large-v3-turbo": 1.00, "qwen3-asr-0.6b": 1.60, "qwen3-asr-1.7b": 1.60}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-from', required=True)
    parser.add_argument('--resume-from', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--profile', default='vyvy372004')
    args = parser.parse_args()
    root = Path.cwd()
    source = checked_path(args.smoke_from).resolve()
    resume = checked_path(args.resume_from).resolve()
    output = checked_path(args.output_dir).resolve()
    if output.exists():
        raise ValueError('Choose a new launcher output directory')
    # Fail before GPU work if the validated 100-sample smoke is not available.
    for slug in MODELS:
        config = load_json(source / slug / 'smoke100/run_config.json')
        if config.get('status') != 'completed' or config['summary']['num_success'] != 100:
            raise ValueError(f'{slug}: requires completed successful smoke100')
    output.mkdir(parents=True)
    statuses, lock, children = {}, threading.Lock(), []

    def run_model(slug, phase, cap, previous):
        total_cost = 0.0
        final = None
        attempts = []
        for attempt in range(1, 4):
            remaining = cap - total_cost
            if remaining <= 0.12:
                break
            job_root = output / f'{slug}-{phase}-attempt{attempt}'
            log_file = output / f'{slug}-{phase}-attempt{attempt}.log'
            command = [sys.executable, '-X', 'utf8', '-m', 'modal', 'run', '--profile', args.profile,
                       '-m', 'asr.modal_suite', '--models', slug, '--phase', phase,
                       '--smoke-from', str(source), '--output-dir', str(job_root),
                       '--budget-usd', str(round(remaining, 4)), '--budget-reserve-usd', '0.1']
            if previous is not None and phase == 'full':
                command += ['--resume-from', str(previous)]
            with log_file.open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                with lock:
                    children.append(child)
                    statuses[f'{slug}/{phase}'] = {'status': 'running', 'attempt': attempt, 'output': str(job_root), 'cap_usd': cap}
                    write_json(output / 'status.json', statuses)
                print(json.dumps({'event': 'model_launched', 'model': slug, 'phase': phase, 'attempt': attempt, 'pid': child.pid}), flush=True)
                code = child.wait()
            summary_file = job_root / 'summary.json'
            if not summary_file.exists():
                final = {'model_slug': slug, 'phase': phase, 'status': 'failed', 'error': f'CLI exited {code}; see {log_file.name}'}
                break
            summary = load_json(summary_file)
            final = summary['jobs'][0]
            total_cost += summary['estimated_compute_usd']
            attempts.append({'root': str(job_root), 'status': final['status'], 'estimated_compute_usd': summary['estimated_compute_usd'], 'error': final.get('error')})
            if final['status'] == 'completed':
                break
            if final['status'].startswith('incomplete'):
                break
            if not any(word in final.get('error', '') for word in ('Connection', 'Conflict', 'RemoteError', 'transport', 'SSL')):
                break
            previous = job_root
            print(json.dumps({'event': 'transport_recovery', 'model': slug, 'phase': phase, 'attempt': attempt, 'reason': final.get('error'), 'retaining_successful_predictions': True}), flush=True)
        final = final or {'model_slug': slug, 'phase': phase, 'status': 'incomplete_budget'}
        final['attempts'] = attempts
        final['estimated_model_compute_usd'] = total_cost
        with lock:
            statuses[f'{slug}/{phase}'] = final
            write_json(output / 'status.json', statuses)
        print(json.dumps({'event': 'model_finished', 'model': slug, 'phase': phase, 'status': final['status'], 'estimated_usd': total_cost}), flush=True)
        return final

    # Threads only supervise subprocesses/files; Modal SDK is never shared by threads.
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_model, slug, 'full', CAPS[slug], resume) for slug in MODELS]
        full = [future.result() for future in as_completed(futures)]
    completed = [r for r in full if r['status'] == 'completed']
    best = sorted(completed, key=lambda r: r['summary']['normalized']['wer'])[:2]
    repeats = []
    if best:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_model, r['model_slug'], 'repeat100', 0.2, None) for r in best]
            repeats = [future.result() for future in as_completed(futures)]
        for config in repeats:
            if config['status'] != 'completed':
                continue
            from metrics.io import read_jsonl
            slug = config['model_slug']
            original_root = Path(next(r for r in full if r['model_slug'] == slug)['attempts'][-1]['root'])
            repeat_root = Path(config['attempts'][-1]['root'])
            original = {r['id']: r['text'] for r in read_jsonl(original_root / slug / 'full/predictions.jsonl')}
            repeated = read_jsonl(repeat_root / slug / 'repeat100/predictions.jsonl')
            repro = {'num_samples': len(repeated), 'same_text': sum(r['text'] == original[r['id']] for r in repeated),
                     'all_match': all(r['status'] == 'ok' and r['text'] == original[r['id']] for r in repeated)}
            write_json(repeat_root / slug / 'repeat100/reproducibility.json', repro)
    report = {'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'full': full, 'repeats': repeats,
              'estimated_new_compute_usd': sum(r['estimated_model_compute_usd'] for r in full + repeats),
              'budget_caps_usd': CAPS, 'prior_run': str(resume)}
    write_json(output / 'summary.json', report)
    print(json.dumps({'event': 'all_models_finished', 'output': str(output), 'completed_full': len(completed),
                      'estimated_new_compute_usd': report['estimated_new_compute_usd']}), flush=True)


if __name__ == '__main__':
    main()
