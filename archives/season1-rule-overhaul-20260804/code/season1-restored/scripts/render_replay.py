"""把保存的审计 JSON 再次渲染成 Markdown。

运行：
  python3 scripts/render_replay.py records/某局.json
  python3 scripts/render_replay.py --public records/某局.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game.replay import (
    project_public_record,
    render_audit_markdown,
    render_public_markdown,
)


def write_private(path: Path, content: str) -> None:
    """手动重渲染也沿用记录文件的 600 权限。"""

    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description="重新渲染狼人杀复盘")
    parser.add_argument("record", type=Path, help="审计 JSON 文件")
    parser.add_argument("--public", action="store_true", help="生成可公开展示的 JSON 和 Markdown")
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))
    if args.public:
        public_record = project_public_record(record)
        public_json = args.record.with_name(args.record.stem + ".public.json")
        public_markdown = args.record.with_name(args.record.stem + ".public.md")
        write_private(public_json, json.dumps(public_record, ensure_ascii=False, indent=2) + "\n")
        write_private(public_markdown, render_public_markdown(public_record))
        print(public_json)
        print(public_markdown)
    else:
        output = args.record.with_suffix(".md")
        write_private(output, render_audit_markdown(record))
        print(output)


if __name__ == "__main__":
    main()
