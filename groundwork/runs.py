"""Durable stage results and usage for a single local run."""
import hashlib
import json
import os
from pathlib import Path

from .errors import ModelError, ModelResponseError

COUNTERS = ('_actual_calls', '_failed_requests', '_input_tokens', '_output_tokens',
            '_known_cost', '_cost_complete', '_tokens_complete', '_by_model', '_proposal_draft')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RunModels:
    """Replay completed stages without paying again; never replay an ambiguous call."""
    def __init__(self, path, signature, factory, resume=False):
        self.path = Path(path)
        self.factory = factory
        self.model = None
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if self.path.exists():
            if not resume:
                raise ValueError('Run exists; use --resume with the original inputs/settings')
            self.state = json.loads(self.path.read_text())
            if self.state['fingerprint'] != fingerprint:
                raise ValueError('Resume inputs/settings differ from the saved run')
        else:
            if resume:
                raise ValueError('No saved run found to resume')
            self.state = {'fingerprint': fingerprint, 'signature': signature, 'calls': {},
                          'usage': {}, 'counters': {}, 'pending': None}
            self.save()
        if self.state['pending']:
            raise ValueError('An interrupted model stage has unknown completion/cost. This run will not '
                             'reissue it and risk duplicate spending. Inspect .run/state.json; use a new '
                             'output directory only if you choose to pay for a fresh run.')

    def save(self):
        atomic_json(self.path, self.state)

    @property
    def usage(self):
        return self.state['usage']

    def call(self, operation, *args, **kwargs):
        key = hashlib.sha256(json.dumps([operation, args, kwargs], sort_keys=True).encode()).hexdigest()
        cached = self.state['calls'].get(key)
        if cached is not None:
            if 'error' in cached:
                error = ModelResponseError if cached['type'] == 'ModelResponseError' else ModelError
                raise error(cached['error'])
            return cached['result']
        if self.model is None:
            self.model = self.factory()
            for field, value in self.state['counters'].items():
                setattr(self.model, field, value)
        self.state['pending'] = {'operation': operation, 'case_id': args[0].get('case_id')}
        self.save()
        try:
            result = getattr(self.model, operation)(*args, **kwargs)
        except ModelError as exc:
            self.state['calls'][key] = {'error': str(exc), 'type': type(exc).__name__}
            self.checkpoint()
            raise
        except Exception:
            # Preserve pending: the remote operation's completion is ambiguous.
            raise
        self.state['calls'][key] = {'result': result}
        self.checkpoint()
        return result

    def checkpoint(self):
        self.state['usage'] = self.model.usage
        self.state['counters'] = {field: getattr(self.model, field) for field in COUNTERS
                                  if hasattr(self.model, field)}
        self.state['pending'] = None
        self.save()

    def propose(self, *args, **kwargs):
        return self.call('propose', *args, **kwargs)

    def review(self, *args, **kwargs):
        return self.call('review', *args, **kwargs)

    def solve(self, *args, **kwargs):
        return self.call('solve', *args, **kwargs)
