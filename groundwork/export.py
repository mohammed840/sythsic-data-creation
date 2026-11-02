"""Training JSONL and isolated Harbor tasks from the same accepted task record."""

import json
from collections import Counter
from pathlib import Path

from . import grading


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_text(task):
    return "\n\n".join(f"[{d['id']}]\n{d['text']}" for d in task["documents"])


def user_prompt(task):
    return (task["instruction"] + "\n\nReturn only a JSON object with exactly this schema: "
            + task["answer_schema"] + "\n\nSources:\n" + source_text(task))


def export_run(records, out, split, usage):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    accepted = [r for r in records if r["accepted"]]
    for name, rows in (("accepted.jsonl", accepted), ("rejected.jsonl", [r for r in records if not r["accepted"]])):
        with (out / name).open("w") as f:
            for row in rows:
                f.write(json.dumps({**row, "split": split}, ensure_ascii=False) + "\n")
    if split == "train":
        with (out / "train.jsonl").open("w") as f:
            for task in accepted:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": user_prompt(task)},
                    {"role": "assistant", "content": json.dumps(task["expected"], ensure_ascii=False)},
                ]}, ensure_ascii=False) + "\n")
    else:
        for task in accepted:
            export_harbor(task, out / "harbor" / task["id"])
    summary = {"split": split, "accepted": len(accepted), "rejected": len(records) - len(accepted),
               "domains": sorted({r["domain"] for r in records}), "usage": usage,
               "ground_truth": "executable_rule", "human_reviewed": False}
    summary['automatic_repairs'] = sum(bool(a.get('format_repair_attempted')) for r in records for a in r.get('attempts', []))
    failures = Counter()
    for row in records:
        if not row['accepted']:
            reason = row.get('error_reason') or '; '.join(
                error for attempt in row.get('attempts', []) for error in attempt.get('errors', []))
            if not reason:
                reason = '; '.join(a.get('review', {}).get('feedback', '') for a in row.get('attempts', []))
            failures[reason or row['review_status']] += 1
    summary['failure_reasons'] = dict(failures)
    write_json(out / "report.json", summary)
    return summary


def export_harbor(task, path):
    for folder in ("environment", "tests", "solution"):
        (path / folder).mkdir(parents=True, exist_ok=False)
    (path / "instruction.md").write_text(
        task["instruction"] + "\n\nRead /app/sources.txt. Write your answer to /app/answer.json.\n"
        "Return exactly this JSON schema, without extra fields: " + task["answer_schema"] + "\n")
    (path / "task.toml").write_text(
        'schema_version = "1.4"\nartifacts = ["/app/answer.json"]\n\n'
        f'[task]\nname = "groundwork/{task["id"]}"\n\n'
        f'[metadata]\ncategory = "{task["domain"]}"\nground_truth = "executable_rule"\n\n'
        '[agent]\ntimeout_sec = 120.0\n\n'
        '[environment]\ncpus = 1\nmemory_mb = 512\nstorage_mb = 1024\n'
        'network_mode = "none"\n\n'
        '[verifier]\ntimeout_sec = 30.0\nenvironment_mode = "separate"\n\n'
        '[verifier.environment]\ncpus = 1\nmemory_mb = 512\nstorage_mb = 1024\nnetwork_mode = "none"\n')
    (path / "environment" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /app\nCOPY sources.txt /app/sources.txt\n")
    (path / "environment" / "sources.txt").write_text(source_text(task))
    (path / "tests" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nWORKDIR /app\nRUN mkdir -p /tests /logs/verifier\nCOPY . /tests/\n")
    write_json(path / "tests" / "expected.json", task["expected"])
    (path / "tests" / "grade.py").write_text(Path(grading.__file__).read_text())
    (path / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\npython /tests/grade.py /tests/expected.json /app/answer.json /logs/verifier/reward.txt\n")
    # A reference replay is for authors/oracle trials, never copied into the agent image.
    answer = json.dumps(task["expected"])
    (path / "solution" / "solve.sh").write_text(
        "#!/bin/sh\nset -eu\nmkdir -p /app\ncat > /app/answer.json <<'ANSWER'\n" + answer + "\nANSWER\n")
    (path / "tests" / "test.sh").chmod(0o755)
    (path / "solution" / "solve.sh").chmod(0o755)
