import argparse
import json
import os
import sys
import tomllib
import math
import shutil
import fcntl
from contextlib import contextmanager
from pathlib import Path

from .domains import load_bundle, make_tasks
from .export import export_run
from .pipeline import OfflineModels, generate
from .runs import RunModels, atomic_json


def register_split(tasks, split, registry):
    """Protect related sources across runs, not random rows after generation."""
    registry = Path(registry)
    previous = json.loads(registry.read_text()) if registry.exists() else {}
    updated = dict(previous)
    for task in tasks:
        for key in ("family:" + task["family"], "source:" + task["source_hash"]):
            if key in previous and previous[key] != split:
                raise ValueError(f"Source already assigned to {previous[key]}; use independent source families for {split}")
            updated[key] = split
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(updated, indent=2) + "\n")


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this run/registry') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description="Generate rule-grounded training data and Harbor evaluation tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-seed", help="Convert text seed files into a reviewable JSON bundle")
    imp.add_argument("--resume", action="store_true")
    imp.add_argument("sources", nargs="+", type=Path)
    imp.add_argument("--domain", choices=("business", "education", "legal", "finance"), required=True)
    imp.add_argument("--family", required=True)
    imp.add_argument("--model")
    imp.add_argument("--config", type=Path, default=Path("groundwork.toml"))
    imp.add_argument("--out", type=Path, required=True)
    imp.add_argument("--env-file", type=Path, default=Path(".env"))
    gen = sub.add_parser("generate")
    gen.add_argument("source", type=Path)
    gen.add_argument("--config", type=Path, default=Path("groundwork.toml"))
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--split", choices=("train", "dev", "eval"), required=True)
    gen.add_argument("--count", type=int, default=2)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--max-revisions", type=int, choices=range(3), default=None)
    gen.add_argument("--offline", action="store_true", help="Template demonstration; no RLM/API calls")
    gen.add_argument("--env-file", type=Path, default=Path(".env"))
    gen.add_argument("--registry", type=Path, default=Path(".groundwork/splits.json"))
    gen.add_argument("--root-model")
    gen.add_argument("--sub-model")
    gen.add_argument("--target-model")
    gen.add_argument("--reviewer-model")
    gen.add_argument("--max-budget", type=float, default=None, help="USD stop threshold; an in-flight call may exceed it")
    gen.add_argument("--max-calls", type=int, default=None, help="Observed-call stop threshold; an RLM completion may exceed it")
    gen.add_argument("--max-iterations", type=int, default=None)
    inspect = sub.add_parser('inspect', help='Show bundle or run report without model calls')
    inspect.add_argument('path', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            path = args.path
            if path.is_dir():
                path = path / ('report.json' if (path / 'report.json').exists() else 'import-report.json')
            if path.name == 'bundle.json' or path.suffix == '.json' and 'report' not in path.name:
                bundle = load_bundle(path)
                print(json.dumps({'domain': bundle['domain'], 'family': bundle['family'],
                                  'cases': len(bundle['cases']), 'policy': bundle['policy']}, indent=2))
            else:
                print(json.dumps(json.loads(path.read_text()), indent=2))
            return 0
        settings = {}
        if args.config.exists():
            settings = tomllib.loads(args.config.read_text()).get('groundwork', {})
            allowed = {'root_model', 'sub_model', 'target_model', 'reviewer_model', 'import_model',
                       'max_budget', 'max_calls', 'max_iterations', 'max_revisions'}
            if set(settings) - allowed:
                raise ValueError('Unknown project configuration field')
        if args.command == "import-seed":
            from dotenv import load_dotenv
            from .seed import import_seed
            load_dotenv(args.env_file, override=False)
            model = args.model or os.getenv('GROUNDWORK_IMPORT_MODEL') or settings.get('import_model')
            if not model:
                raise ValueError('Set --model or import_model in groundwork.toml')
            if args.out.exists() and not args.resume:
                raise ValueError('Output exists; use --resume or a fresh directory')
            with run_lock(args.out / '.run/lock'):
                report = import_seed(args.sources, args.domain, args.family, args.out, model, resume=args.resume, locked=True)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "needs_review" else 1
        for key, default in dict(max_budget=1.0, max_calls=40, max_iterations=4, max_revisions=0).items():
            if getattr(args, key) is None:
                setattr(args, key, settings.get(key, default))
        if args.out.exists() and not args.resume:
            raise ValueError("Output already exists; use --resume or a fresh directory")
        if type(args.max_revisions) is not int or args.max_revisions not in range(3):
            raise ValueError('max_revisions must be 0, 1, or 2')
        if any(type(getattr(args, key)) is not int for key in ('max_calls', 'max_iterations')):
            raise ValueError('Call and iteration limits must be integers')
        if type(args.max_budget) not in (int, float) or not math.isfinite(args.max_budget):
            raise ValueError('Budget must be a finite positive number')
        if args.max_budget <= 0 or args.max_calls < 1 or args.max_iterations < 1:
            raise ValueError("Budget, calls, and iterations must be positive")
        tasks = list(make_tasks(load_bundle(args.source), args.count, args.seed))
        if not args.offline:
            if args.split == "eval" and args.max_revisions != 0:
                raise ValueError("Final eval generation requires --max-revisions 0; tune on dev, then freeze eval")
            from dotenv import load_dotenv
            load_dotenv(args.env_file, override=False)
            from .models import Models
            config = {key: getattr(args, key) or os.getenv("GROUNDWORK_" + key.upper()) or settings.get(key)
                      for key in ("root_model", "sub_model", "target_model", "reviewer_model")}
            if not all(config.values()):
                raise ValueError("Set all four model roles using flags or GROUNDWORK_*_MODEL environment variables")
            config.update(max_budget=args.max_budget, max_calls=args.max_calls,
                          max_iterations=args.max_iterations, environment="docker")
            factory = lambda: Models(config)
        else:
            config = {"mode": "offline"}
            factory = OfflineModels
        signature = {'source_hash': tasks[0]['source_hash'], 'family': tasks[0]['family'],
                     'split': args.split, 'count': args.count, 'seed': args.seed,
                     'max_revisions': args.max_revisions, 'config': config, 'runtime_version': 1,
                     'limits': {k: getattr(args, k) for k in ('max_budget','max_calls','max_iterations')}}
        with run_lock(args.out / '.run/lock'):
            models = RunModels(args.out / '.run/state.json', signature, factory, args.resume)
            with run_lock(args.registry.with_suffix('.lock')):
                register_split(tasks, args.split, args.registry)
            records = []
            for task in tasks:
                row_path = args.out / '.run' / (task['id'] + '.json')
                if row_path.exists():
                    row = json.loads(row_path.read_text())
                else:
                    row = generate(task, models, 0 if args.offline else args.max_revisions)
                    # Ambiguous remote calls remain pending and cannot be silently replayed.
                    if models.state['pending']:
                        raise ValueError('Model stage interrupted with unknown completion; see .run/state.json')
                    if args.offline:
                        row["review_status"] = "offline_template_unreviewed"
                    row["generation_mode"] = "offline_template" if args.offline else "rlm_autodata"
                    atomic_json(row_path, row)
                records.append(row)
                print(f"{task['case_id']}: {'accepted' if row['accepted'] else 'rejected'}", file=sys.stderr)
            # Rebuild exports in private staging, preserving stage checkpoints on interruption.
            staging = args.out / '.run/export'
            if staging.exists():
                shutil.rmtree(staging)
            report = export_run(records, staging, args.split, models.usage)
            report['source'] = {'family': tasks[0]['family'], 'sha256': tasks[0]['source_hash'],
                                'path': str(args.source)}
            report['run_settings'] = signature
            atomic_json(staging / 'report.json', report)
            for file in staging.iterdir():
                destination = args.out / file.name
                if destination.is_dir():
                    shutil.rmtree(destination)
                file.replace(destination)
            print(json.dumps(report, indent=2))
            return 0 if report['accepted'] == len(records) else 1

    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
