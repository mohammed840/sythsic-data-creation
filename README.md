# Groundwork Data

Local CLI for creating rule-grounded training data and Harbor eval tasks from your own seed files. Supports business refunds, education eligibility, legal contract deadlines, and finance invoice calculations. Uses RLM, a bounded AutoData-inspired loop, and OpenRouter.

This repository includes runtime code and setup only. Supply your own data locally.

## Install

Requires Python 3.11+ on macOS/Linux. Live RLM generation and Harbor need Docker.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Model roles and limits are in `groundwork.toml`; Luna is selected by default. Keep local inputs in `my-data/`, which Git ignores.

## Use

```sh
.venv/bin/groundwork import-seed my-data/policy.md my-data/records.csv --domain finance --family my-training-v1 --out outputs/seed
.venv/bin/groundwork inspect outputs/seed
.venv/bin/groundwork inspect outputs/seed/bundle.json
```

Review the extracted bundle against every source fact. It is structurally validated but not human-reviewed.

```sh
.venv/bin/groundwork generate outputs/seed/bundle.json --split train --count 3 --out outputs/training
.venv/bin/groundwork inspect outputs/training
```

Use `--split eval` or `dev` for Harbor task exports. Use independent seed policies, records and source families for training and evaluation. The split registry refuses related sources crossing splits. Add `--resume` to the exact original command to reuse completed stages without new calls. Ambiguous in-flight requests stop rather than risk duplicate billing. Rejected completed tasks stay rejected on resume.

Supported inputs: UTF-8 MD, CSV, TXT, TSV, JSON and JSONL; up to 20 files and 200 KB combined. One task per selected case; `--count` cannot exceed case count. Missing facts and incompatible rules are reported rather than invented.

| Domain | Supported facts and answer |
|---|---|
| business | Return-window/cap policy, order price/day/unused/receipt; eligibility and refund |
| education | Score threshold and required/completed modules; eligibility and missing modules |
| legal | Explicit reporting date, service/response dates, business-day period, holidays and extension; deadline and timely/overdue status |
| finance | Quantity/price, discount, approved credit, paid/void flags, tax bps and FX rates; payment status and USD cents |

Code computes answers from reviewed rules. Legal uses Monday–Friday excluding explicit holidays, with the service date excluded. Finance rounds tax half up per invoice, then converts to USD and rounds half up again. Unsupported semantics are outside this release. No real-world legal or financial rules are inferred.

Training exports are `train.jsonl`; eval exports are under `harbor/`. Author reports and `.run/` journals contain source facts and answer keys; keep all outputs private. Reports show accepted/rejected counts, reasons, repairs, provenance and known costs. Exit code 1 means partial/rejected results; code 2 means input/config error.

## Harbor

```sh
.venv/bin/python -m pip install '.[harbor]'
.venv/bin/harbor run -p outputs/evaluation/harbor -a oracle --env docker --jobs-dir outputs/jobs
```

First generate `outputs/evaluation` using independent eval seeds. Oracle should score 1; `-a nop` should score 0. The included `groundwork.harbor_agent:OpenRouterAgent` adapter runs a single OpenRouter model answer without repair. Never expose the author export root or verifier keys to evaluated agents.

Model budget/call limits are stop thresholds, not hard spending ceilings. In-flight RLM work can exceed them, and timeouts cannot interrupt every Docker operation. Unknown costs stop later paid requests. Use account-side spending controls where available. No fine-tuning or arbitrary-document ground-truth guarantees are included.

MIT licensed. Dependencies retain their own licenses.
