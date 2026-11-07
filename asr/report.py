"""Build comparison artifacts after all five full-train jobs finish."""

import argparse
import csv
import hashlib
from pathlib import Path

from asr.registry import MODELS
from metrics.io import checked_path, load_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--update-results', action='store_true')
    parser.add_argument('--billing-after')
    args = parser.parse_args()
    folder = checked_path(args.run_dir).resolve()
    report = load_json(folder / 'summary.json')
    jobs = {c['model_slug']: c for c in report['full']}
    if set(jobs) != set(MODELS) or any(c['status'] != 'completed' for c in jobs.values()):
        raise ValueError('Comparison requires five completed full-train jobs')
    rows, manifest_hashes = [], set()
    for slug, config in jobs.items():
        attempt_root = Path(config['attempts'][-1]['root'])
        provenance = load_json(attempt_root / 'provenance.json')
        manifest_hashes.add(provenance['manifest_sha256'])
        predictions_file = attempt_root / slug / 'full/predictions.jsonl'
        predictions = read_jsonl(predictions_file)
        if len(predictions) != 1545 or len({r['id'] for r in predictions}) != 1545:
            raise ValueError(f'{slug}: incomplete or duplicate predictions')
        summary = config['summary']
        rows.append({
            'model': slug, 'scope': 'full_train_20261107', 'num_samples': 1545,
            'num_predictions': len(predictions), 'num_parameters': config['num_parameters'],
            'gpu': config['gpu'], 'raw_wer_percent': summary['raw']['wer_percent'],
            'raw_cer_percent': summary['raw']['cer_percent'],
            'wer_percent': summary['normalized']['wer_percent'],
            'cer_percent': summary['normalized']['cer_percent'],
            'exact_match_rate': summary['normalized']['exact_match_rate'],
            'mean_seconds': summary['latency_seconds']['mean'],
            'p95_seconds': summary['latency_seconds']['p95'],
            'max_seconds': summary['latency_seconds']['max'],
            'real_time_factor': summary['real_time_factor'],
            'peak_pytorch_mib': max(r.get('peak_gpu_memory_mb', 0) for r in predictions),
            'num_errors': summary['num_errors'], 'num_timeouts': summary['num_timeouts'],
            'hit_token_limit_count': config['hit_token_limit_count'],
            'num_reused_predictions': config.get('num_reused_predictions', 0),
            'estimated_isolated_compute_usd': config['estimated_model_compute_usd'],
            'cost_scope': 'isolated attempts only; earlier shared-client attempts excluded',
            'manifest_sha256': provenance['manifest_sha256'],
            'predictions': str(predictions_file.relative_to(folder)),
        })
    if len(manifest_hashes) != 1:
        raise ValueError('Models were not evaluated on identical input manifests')
    rows.sort(key=lambda r: r['wer_percent'])
    write_json(folder / 'comparison.json', {'models': rows, 'repeats': report['repeats']})
    with (folder / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline_file = Path('asr/RESULTS.md')
    snapshot = folder / 'historical_phowhisper_small.md'
    historical = snapshot.read_text(encoding='utf-8') if snapshot.exists() else baseline_file.read_text(encoding='utf-8').split('## So sánh các baseline ASR — 07/11/2026')[0].rstrip() + '\n'
    snapshot.write_text(historical, encoding='utf-8')
    write_json(folder / 'historical_source.json', {
        'source': str(baseline_file.resolve()), 'kind': 'documentation_only',
        'sha256': hashlib.sha256(historical.encode('utf-8')).hexdigest(),
        'note': 'Original PhoWhisper-small predictions are not present in the D repo; input hash cannot be verified for this historical row.',
    })
    lines = ['# So sánh baseline ASR — 07/11/2026', '',
             '5 model mới đã chạy đủ 1.545 audio yêu cầu cuối của train, cùng manifest, T4, FP16, greedy và seed 42. Nhãn chỉ dùng local để chấm; worker inference chặn network.', '',
             '| Model | Tham số thực | WER chuẩn hóa | CER chuẩn hóa | Mean / p95 (giây) | Lỗi / timeout |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['num_parameters']:,} | {row['wer_percent']:.2f}% | {row['cer_percent']:.2f}% | {row['mean_seconds']:.3f} / {row['p95_seconds']:.3f} | {row['num_errors']} / {row['num_timeouts']} |")
    lines += ['| PhoWhisper-small, baseline cũ* | 241,734,912 | 34.20% | 19.49% | 0.970 / 1.607 | 0 / 0 |', '',
              '*Dòng PhoWhisper-small trích báo cáo ngày 06/11, không phải một job mới. Prediction gốc chưa có trong repo D nên chưa xác minh được input hash của dòng lịch sử. Bản nguồn được lưu tại `historical_phowhisper_small.md`.', '',
              'WER/CER tính corpus với cùng phiên bản metrics cho 5 model mới, gồm raw và normalized trong CSV/JSON. Normalized giữ dấu tiếng Việt, số, dấu âm và dấu phân cách số; không chuyển số thành chữ. CER tính cả khoảng trắng.', '',
              'Latency đo riêng WAV decode + preprocess + generation + text decode, không gồm load/warmup/RPC. Các prefix full hợp lệ từ lượt v3 được giữ với timing đã đo và source hash không đổi; config ghi `resumed_from`, số prediction tái sử dụng và các attempt. Không gán latency bằng 0 cho dữ liệu được tiếp tục.', '',
              'Số tham số lấy từ `sum(p.numel() for p in model.parameters())`, tính trọng số chia sẻ một lần. Tổng tensor trên Hugging Face có cách đếm khác và được lưu riêng cho Qwen. Ghép ASR + LLM vẫn cần kiểm tra tổng <6B, LLM <5B, context ≤16.384 token và thời gian toàn pipeline trên A30.', '',
              '## Kiểm tra tái lập', '']
    for config in report['repeats']:
        if config['status'] == 'completed':
            repeat_root = Path(config['attempts'][-1]['root'])
            result = load_json(repeat_root / config['model_slug'] / 'repeat100/reproducibility.json')
            lines.append(f"- {config['model_slug']}: {result['same_text']}/{result['num_samples']} transcript giống lượt full; all_match={result['all_match']}.")
        else:
            lines.append(f"- {config['model_slug']}: {config['status']}.")
    lines += ['', '## Chi phí và artifact', '',
              f"Compute ước tính của launcher (full tiếp tục + repeat, chưa gồm các lượt trước): **{report['estimated_new_compute_usd']:.3f} USD**. Các cap cộng 7,65 USD cho full và 0,40 USD cho repeat. Chi phí theo model trong CSV chỉ tính các attempt ở launcher, không phải toàn bộ chi phí từ đầu smoke.", '']
    if args.billing_after:
        billing = load_json(args.billing_after)
        before = load_json(folder.parent / 'billing_before_20261107.json')
        credit_now = abs(float(billing['adjustments']['credits']))
        delta = credit_now - float(before['credits_applied_usd'])
        lines += [f"Snapshot billing workspace: credit đã dùng tháng 11 là **{credit_now:.2f} USD**, tăng **{delta:.2f} USD** từ mốc trước đợt chạy; billed_cost **{float(billing['billed_cost']):.2f} USD**. Chênh lệch gồm các smoke/khởi chạy đã dừng và có thể cập nhật trễ hoặc chứa workload khác; không phải hóa đơn tách riêng app.", '']
    lines += ['- `comparison.csv` / `comparison.json`: 5 model mới, thông số đầy đủ và đường dẫn prediction.',
              '- `summary.json`: kết quả tất cả job/attempt, repeat và compute ước tính.',
              '- Các thư mục `<model>-<phase>-attemptN/`: manifest, config, metrics, predictions và provenance.', '',
              'Đây là benchmark trên train, chưa có kết quả holdout, ASR của history hoặc toàn pipeline ASR + LLM + tool loop trên A30.']
    markdown = '\n'.join(lines) + '\n'
    (folder / 'report.md').write_text(markdown, encoding='utf-8')
    if args.update_results:
        section = markdown.replace('# So sánh baseline ASR — 07/11/2026', '## So sánh các baseline ASR — 07/11/2026', 1)
        section = section.replace('\n## Kiểm tra tái lập', '\n### Kiểm tra tái lập').replace('\n## Chi phí và artifact', '\n### Chi phí và artifact')
        relative = folder.relative_to(Path.cwd())
        section += f'\nArtifact lượt mới: [report](../{relative.as_posix()}/report.md), [CSV](../{relative.as_posix()}/comparison.csv).\n'
        archived = historical.replace('\n## ', '\n### ')
        archived = archived.replace('# PhoWhisper-small: baseline train ngày 06/11/2026', '## Chi tiết PhoWhisper-small — 06/11/2026', 1)
        baseline_file.write_text('# Kết quả baseline ASR\n\n' + section + '\n---\n\n' + archived, encoding='utf-8')
    print(f'Comparison written: {folder / "report.md"}')


if __name__ == '__main__':
    main()
