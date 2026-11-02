"""Single-call OpenRouter solver for Groundwork's Harbor tasks (requires Harbor)."""
import base64
import json
import os

from openai import AsyncOpenAI
from harbor.agents.base import BaseAgent


class OpenRouterAgent(BaseAgent):
    @staticmethod
    def name():
        return 'groundwork-openrouter'

    def version(self):
        return '0.1.0'

    async def setup(self, environment):
        if not self.model_name or not os.getenv('OPENROUTER_API_KEY'):
            raise ValueError('Model name and OPENROUTER_API_KEY are required')

    async def run(self, instruction, environment, context):
        sources = await environment.exec(command='cat /app/sources.txt')
        if sources.return_code != 0:
            raise RuntimeError('Unable to read task sources')
        messages = [
            {'role': 'system', 'content': 'Solve the task using only the provided source facts. '
             'Return only the requested JSON answer. The harness will save it to /app/answer.json.'},
            {'role': 'user', 'content': instruction + '\n\nSources:\n' + sources.stdout},
        ]
        async with AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                               base_url='https://openrouter.ai/api/v1', timeout=90,
                               max_retries=0) as client:
            try:
                response = await client.chat.completions.create(
                    model=self.model_name, messages=messages, max_tokens=8192,
                    response_format={'type': 'json_object'},
                )
            except Exception as exc:
                raise RuntimeError(f'OpenRouter request failed ({type(exc).__name__})') from None
        usage = response.usage
        if usage:
            context.n_input_tokens = usage.prompt_tokens
            context.n_output_tokens = usage.completion_tokens
            context.cost_usd = getattr(usage, 'cost', None)
        choice = response.choices[0]
        content = choice.message.content or ''
        context.metadata = {'protocol': 'single_call_no_repair', 'finish_reason': choice.finish_reason}
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / 'response.json').write_text(json.dumps({
            'model': self.model_name, 'messages': messages, 'response': content,
            'finish_reason': choice.finish_reason,
        }, indent=2))
        # Preserve the model's answer exactly; Harbor owns parsing and scoring.
        encoded = base64.b64encode(content.encode()).decode()
        result = await environment.exec(command=
            "python -c \"import base64; from pathlib import Path; "
            f"Path('/app/answer.json').write_bytes(base64.b64decode('{encoded}'))\"")
        if result.return_code != 0:
            raise RuntimeError('Unable to write model answer')
