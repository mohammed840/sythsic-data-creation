"""Four deliberately small rule families; no model-generated answer keys."""

import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_bundle(path):
    bundle = json.loads(Path(path).read_text())
    return validate_bundle(bundle)


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("Seed bundle must be an object")
    if bundle.get("domain") not in ("business", "education", "legal", "finance"):
        raise ValueError("domain must be business, education, legal, or finance")
    if not isinstance(bundle.get("family"), str) or not bundle["family"].strip():
        raise ValueError("A stable source family is required")
    if not isinstance(bundle.get("policy"), dict):
        raise ValueError("policy must be an object")
    if not isinstance(bundle.get("cases"), list) or not bundle["cases"]:
        raise ValueError("At least one source case is required")
    if any(not isinstance(case, dict) for case in bundle["cases"]):
        raise ValueError("Each case must be an object")
    ids = [case.get("id") for case in bundle["cases"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Cases require unique nonempty string IDs")
    for case in bundle["cases"]:
        answer_for(bundle, case)
    return bundle


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def iso_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Dates must use YYYY-MM-DD')
    return date.fromisoformat(value)


def legal_result(case, policy):
    deadline = iso_date(case['served'])
    days = policy['business_days'] + (policy['extension_days'] if case['signed_extension'] else 0)
    while days:
        deadline += timedelta(days=1)
        if deadline.weekday() < 5 and deadline.isoformat() not in policy['holidays']:
            days -= 1
    response = iso_date(case['response_received']) if case['response_received'] is not None else None
    timely = response is not None and response <= deadline
    return {'deadline': deadline.isoformat(), 'timely': timely,
            'overdue': not timely and deadline < iso_date(policy['as_of'])}


def finance_result(case, policy):
    payable = not case['void'] and not case['paid']
    credit = case['credit_cents'] if case['credit_approved'] else 0
    net = max(0, case['unit_cents'] * case['quantity'] - case['discount_cents'] - credit)
    tax = (net * case['tax_bps'] + 5000) // 10000
    usd = ((net + tax) * policy['fx_millionths'][case['currency']] + 500000) // 1000000
    return {'payable': payable, 'usd_cents': usd if payable else 0}


def answer_for(bundle, case):
    policy = bundle["policy"]
    if bundle["domain"] == "business":
        window = integer(policy["return_window_days"], "return_window_days")
        cap = integer(policy["refund_cap_cents"], "refund_cap_cents")
        days = integer(case["days_since_delivery"], "days_since_delivery")
        price = integer(case["price_cents"], "price_cents")
        for field in ("unused", "receipt_present"):
            if type(case[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        eligible = days <= window and case["unused"] and case["receipt_present"]
        return {"eligible": eligible, "refund_cents": min(price, cap) if eligible else 0}
    if bundle['domain'] == 'legal':
        for field in ('business_days', 'extension_days'):
            if integer(policy[field], field) > 366:
                raise ValueError(f'{field} must be at most 366 for this recipe')
        if policy['business_days'] == 0:
            raise ValueError('business_days must be positive')
        holidays = policy['holidays']
        if not isinstance(holidays, list):
            raise ValueError('holidays must be an explicit array, possibly empty')
        for holiday in holidays:
            iso_date(holiday)
        served, as_of = iso_date(case['served']), iso_date(policy['as_of'])
        if served > as_of:
            raise ValueError('served must be on or before as_of')
        if type(case['signed_extension']) is not bool:
            raise ValueError('signed_extension must be boolean')
        response = case['response_received']
        if response is not None and not served <= iso_date(response) <= as_of:
            raise ValueError('response_received must be null or between served and as_of')
        return legal_result(case, policy)
    if bundle['domain'] == 'finance':
        rates = policy['fx_millionths']
        if not isinstance(rates, dict) or not rates:
            raise ValueError('Explicit fx_millionths mapping is required')
        for currency, rate in rates.items():
            if not isinstance(currency, str) or not currency or integer(rate, 'FX rate') == 0:
                raise ValueError('Currency names and positive integer FX rates are required')
        if case['currency'] not in rates:
            raise ValueError('Missing FX rate for invoice currency')
        for field in ('unit_cents', 'quantity', 'discount_cents', 'credit_cents', 'tax_bps'):
            integer(case[field], field)
        for field in ('credit_approved', 'void', 'paid'):
            if type(case[field]) is not bool:
                raise ValueError(f'{field} must be boolean')
        return finance_result(case, policy)
    minimum = integer(policy["minimum_score"], "minimum_score")
    required = policy["required_modules"]
    completed = case["completed_modules"]
    if not isinstance(required, list) or not isinstance(completed, list):
        raise ValueError("Module lists must be arrays")
    if any(not isinstance(m, str) for m in required + completed):
        raise ValueError("Module names must be strings")
    if len(set(required)) != len(required) or len(set(completed)) != len(completed):
        raise ValueError("Module names must be unique")
    score = integer(case["score"], "score")
    missing = sorted(set(required) - set(completed))
    return {"eligible": score >= minimum and not missing, "missing_modules": missing}


def make_tasks(bundle, count, seed):
    if not 1 <= count <= len(bundle["cases"]):
        raise ValueError(f"count must be between 1 and {len(bundle['cases'])}; add distinct source cases to scale")
    fingerprint = hashlib.sha256(canonical({k: v for k, v in bundle.items() if k != "family"}).encode()).hexdigest()
    cases = random.Random(seed).sample(bundle["cases"], count)
    policy = bundle["policy"]
    if bundle["domain"] == "business":
        rule = (f"A refund is eligible if and only if delivery was at most {policy['return_window_days']} days ago "
                "(inclusive), the item is unused, and a receipt is present. "
                f"For eligible orders, refund the smaller of price_cents and {policy['refund_cap_cents']}. "
                "Otherwise refund zero cents. All money values are integer cents.")
        question = "Determine refund eligibility and refund amount for order"
        schema = '{"eligible": boolean, "refund_cents": integer}'
    elif bundle["domain"] == "education":
        rule = (f"A student is eligible if and only if score is at least {policy['minimum_score']} "
                f"and every required module is completed. Required modules: {canonical(policy['required_modules'])}. "
                "Report all missing required modules, irrespective of score. Module order does not matter.")
        question = "Determine course eligibility and missing required modules for student"
        schema = '{"eligible": boolean, "missing_modules": array of strings}'
    elif bundle['domain'] == 'legal':
        rule = (f"Use only these supplied contract rules, not external law. Deadline is {policy['business_days']} "
                f"business days AFTER served, excluding the served date. If signed_extension=true, add "
                f"{policy['extension_days']} business days. Business days are Monday–Friday except "
                f"these dates: {canonical(policy['holidays'])}. Use the Gregorian calendar. "
                f"Report as of {policy['as_of']}. Timely means response_received is not null and on or "
                "before the deadline. Overdue means no timely response and deadline is strictly before "
                "as_of; a late response remains overdue. Return deadline as YYYY-MM-DD and both booleans. "
                "All records are complete current snapshots; no other rules apply.")
        question = 'Determine the response deadline and timely/overdue status for notice'
        schema = '{"deadline": "YYYY-MM-DD", "timely": boolean, "overdue": boolean}'
    else:
        rule = ("Use only these supplied accounting rules. Each invoice is a complete current snapshot. "
                "Payable iff void=false AND paid=false, even for a zero balance. All amounts are integer "
                "cents in the invoice currency. credit=credit_cents if credit_approved else 0; "
                "net=max(0,unit_cents*quantity-discount_cents-credit); "
                "tax=floor((net*tax_bps+5000)/10000); gross=net+tax. "
                "usd_cents=floor((gross*rate+500000)/1000000). "
                f"Explicit rate integers (millionths): {canonical(policy['fx_millionths'])}. "
                "Both rounding steps are per invoice, nearest cent with halves upward. "
                "For void or paid invoices return payable=false and usd_cents=0. No other rules apply.")
        question = 'Determine whether payment is required and its USD amount in cents for invoice'
        schema = '{"payable": boolean, "usd_cents": integer}'
    documents = [
        {"id": "policy", "text": rule},
        {"id": "cases", "text": json.dumps(bundle["cases"], ensure_ascii=False, indent=2)},
    ]
    for case in cases:
        task_id = hashlib.sha256((fingerprint + case["id"]).encode()).hexdigest()[:16]
        yield {
            "id": task_id, "domain": bundle["domain"], "family": bundle["family"],
            "source_hash": fingerprint, "case_id": case["id"], "documents": documents,
            "instruction": f"{question} {case['id']}. Use the supplied policy and case records.",
            "answer_schema": schema, "expected": answer_for(bundle, case),
            "evidence": [{"document_id": "policy", "quote": rule},
                         {"document_id": "cases", "case_id": case["id"]}],
            "ground_truth": "executable_rule", "recipe_version": "0.1.0",
        }
