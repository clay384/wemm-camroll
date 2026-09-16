"""Small real Qwen/WeMM interface checks; not an answer-quality benchmark."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from camroll_agent.agent import Agent
from camroll_agent.library import Library
from camroll_agent.qwen import QwenClient
from camroll_agent.wemm import EmbeddingClient


class RecordingQwen(QwenClient):
    def __init__(self):
        super().__init__()
        self.transmissions = []
        self.current_role = None

    def generate_json(self, prompt, image_paths, schema, *, role='caption', max_attempts=3):
        self.transmissions.append({'role': role, 'image_ids': ['img_' + Path(p).parent.name for p in image_paths],
                                   'image_count': len(image_paths)})
        return super().generate_json(prompt, image_paths, schema, role=role, max_attempts=max_attempts)

    def chat(self, messages, tools=None, **kwargs):
        count = sum(1 for m in messages for item in (m.get('content') or [])
                    if isinstance(item, dict) and item.get('type') == 'image_url')
        self.transmissions.append({'role': kwargs.get('role', 'agent'), 'wire_image_count': count})
        return super().chat(messages, tools, **kwargs)


def main():
    memory, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ['M1.json', 'M2.json', 'M3.json', 'M4.json', 'summary.json']):
        raise SystemExit('Probe output exists; use a new run directory before starting charged work')
    embedding = EmbeddingClient()
    question = '请先用search查找相册中的动物照片，再用view_image查看最相关的一张，描述画面中能直接观察到的内容，并引用图片ID。'
    overview = []
    for config in ['M1', 'M2', 'M3', 'M4']:
        target = output / (config + '.json')
        if target.exists():
            raise SystemExit('Probe output exists; use a new run directory to avoid overwriting charged work')
        qwen = RecordingQwen()
        lib = Library(memory, embedding, config=config)
        result = Agent(str(memory), llm=qwen, vlm=qwen, library=lib, config=config).ask(question)
        value = asdict(result)
        value['transmissions'] = qwen.transmissions
        selected = [row for row in qwen.transmissions if row['role'] == 'final' and 'image_ids' in row]
        final_wire = [row['wire_image_count'] for row in qwen.transmissions if row['role'] == 'final' and 'wire_image_count' in row]
        checks = {'answer_returned': bool(result.final_text), 'normal_stop': result.stopped_reason == 'ok',
                  'search_called': any(t['tool'] == 'search' and 'error' not in t['result'] for t in result.tool_trace),
                  'view_called': any(t['tool'] == 'view_image' and 'error' not in t['result'] for t in result.tool_trace),
                  'refs_belong_to_album': bool(result.evidence_image_ids) and set(result.evidence_image_ids) <= set(lib.images),
                  'expected_final_mode': (not selected if config in ['M1', 'M3'] else
                                          len(selected) == 1 and 1 <= selected[0]['image_count'] <= 6 and
                                          bool(final_wire) and all(n == selected[0]['image_count'] for n in final_wire) and
                                          set(result.evidence_image_ids) <= set(selected[0]['image_ids']))}
        value['checks'] = checks
        value['question'] = question
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        overview.append({'config': config, 'checks': checks, 'usage': result.usage,
                         'stopped_reason': result.stopped_reason, 'latency_s': result.latency_s})
        print(json.dumps(overview[-1], ensure_ascii=False), flush=True)
    report = {'purpose': 'interface and image transmission, not answer-quality scoring',
              'checks_passed': all(all(row['checks'].values()) for row in overview),
              'configs': overview, 'budget': qwen.usage_summary()}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['checks_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
