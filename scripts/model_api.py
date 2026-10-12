"""Send one bounded research or editorial request to its configured model.

Request builders keep one shape (instructions, input, max_output_tokens,
reasoning.effort, an optional json_schema format and at most one web_search
tool), so budget bounds and archived request packets do not depend on the
provider. A model whose name starts with "claude-" is sent to Anthropic's
Messages API, and its reply is returned in the same response fields the
callers already validate: status, output text, search sources and usage.
Switching providers is a change of configured model name; any other name
keeps the existing OpenAI path in each caller.

There are no automatic retries: a timeout may follow a billable request.
"""
import json
import os
import re

ANTHROPIC_KEY = 'ANTHROPIC_API_KEY'
# Basic search: one direct search whose results go straight to the model, so
# the single-search reservation in research_discovery.bounds still applies.
SEARCH_TOOL = 'web_search_20250305'
EFFORTS = {'minimal': 'low', 'none': 'low', 'low': 'low', 'medium': 'medium', 'high': 'high',
           'xhigh': 'xhigh', 'max': 'max'}
# Claude's JSON outputs reject these keywords; validate_shape checks them locally.
UNSUPPORTED = ('minLength', 'maxLength', 'maxItems', 'minimum', 'maximum', 'multipleOf')
# Search-result payloads and thinking signatures are opaque and only needed to
# continue a conversation. Each request here is a single isolated turn.
OPAQUE = ('encrypted_content', 'encrypted_index', 'signature')


class NotBilled(RuntimeError):
    """The request was never sent, or the API refused it with a 4xx; nothing was billed."""


def is_claude(model):
    return str(model).startswith('claude-')


def key_name(model):
    return ANTHROPIC_KEY if is_claude(model) else 'OPENAI_API_KEY'


def has_key(model):
    return bool(os.getenv(key_name(model)))


def supported_schema(schema):
    """Drop length/count limits Claude cannot enforce; keep types, enums and required."""
    if isinstance(schema, list):
        return [supported_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    result = {k: supported_schema(v) for k, v in schema.items() if k not in UNSUPPORTED}
    if result.get('minItems', 0) > 1:
        result.pop('minItems')
    return result


def claude_params(request):
    tools = request.get('tools') or []
    schema = ((request.get('text') or {}).get('format') or {}).get('schema')
    system = request['instructions']
    effort = EFFORTS.get((request.get('reasoning') or {}).get('effort'), 'low')
    params = dict(model=request['model'], max_tokens=request['max_output_tokens'],
                  messages=[{'role': 'user', 'content': request['input']}],
                  output_config={'effort': effort})
    if tools:
        if len(tools) != 1 or tools[0].get('type') != 'web_search':
            raise ValueError('Only one web_search tool is supported')
        search = dict(type=SEARCH_TOOL, name='web_search', max_uses=request.get('max_tool_calls', 1))
        domains = (tools[0].get('filters') or {}).get('allowed_domains')
        if domains:
            search['allowed_domains'] = list(domains)
        params['tools'] = [search]
        if request.get('tool_choice') == 'required':
            system += '\nRun the web search before answering.'
        if schema:
            # Search results always carry citations, which cannot be combined
            # with Claude's JSON outputs. Ask for the schema; callers validate it.
            system += ('\nReturn only one JSON object, with no Markdown fences or other text, '
                       'that satisfies this JSON Schema:\n'+json.dumps(schema, separators=(',', ':')))
    elif schema:
        params['output_config']['format'] = dict(type='json_schema', schema=supported_schema(schema))
    params['system'] = system
    return params


def report_failure(stage, error):
    """Put the API's status line in the job log; other errors stay type-only."""
    if isinstance(error, RuntimeError):
        print(f'::warning::{stage} request failed: {error}', flush=True)


def strip_opaque(value):
    if isinstance(value, list):
        return [strip_opaque(v) for v in value]
    if isinstance(value, dict):
        return {k: strip_opaque(v) for k, v in value.items() if k not in OPAQUE}
    return value


def json_text(text):
    """Searched answers are prompted for JSON; return only the object."""
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = text.find('{'), text.rfind('}')
    return text[start:end+1] if 0 <= start < end else text


def from_claude(message, request):
    blocks = message.get('content') or []
    tool_blocks = [i for i, b in enumerate(blocks) if b.get('type') not in ('text', 'thinking', 'redacted_thinking')]
    # Text written before or between searches is not the answer.
    answer = blocks[tool_blocks[-1]+1:] if tool_blocks else blocks
    content, output = [], []
    for block in answer:
        if block.get('type') == 'text':
            notes = [dict(type='url_citation', url=c['url'], title=c.get('title'))
                     for c in block.get('citations') or [] if c.get('type') == 'web_search_result_location' and c.get('url')]
            content.append(dict(type='output_text', text=block['text'], annotations=notes))
    for block in blocks:
        if block.get('type') == 'web_search_tool_result' and isinstance(block.get('content'), list):
            output.append(dict(type='web_search_call', status='completed', action=dict(type='search', sources=[
                dict(type='url', url=r['url']) for r in block['content'] if r.get('type') == 'web_search_result' and r.get('url')])))
    if request.get('tools') and ((request.get('text') or {}).get('format') or {}).get('schema') and content:
        text = json_text(''.join(c['text'] for c in content))
        content = [dict(type='output_text', text=text, annotations=[a for c in content for a in c['annotations']])]
    stop = message.get('stop_reason')
    if stop == 'refusal':
        content.append(dict(type='refusal', refusal=((message.get('stop_details') or {}).get('category') or 'refusal')))
    output.append(dict(type='message', role='assistant', status='completed', content=content))
    raw = message.get('usage') or {}
    usage = dict(input_tokens=sum(raw.get(k) or 0 for k in ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')),
                 output_tokens=raw.get('output_tokens') or 0,
                 web_search_requests=(raw.get('server_tool_use') or {}).get('web_search_requests') or 0)
    response = dict(id=message.get('id'), object='response', provider='anthropic', model=message.get('model'),
                    status='completed' if stop == 'end_turn' else 'incomplete', stop_reason=stop,
                    output=output, usage=usage, anthropic_message=strip_opaque(message))
    if response['status'] != 'completed':
        response['incomplete_details'] = dict(reason=stop)
    return response


def _send(params, timeout):
    import anthropic  # Only Claude runs need the SDK installed.
    client = anthropic.Anthropic(max_retries=0, timeout=anthropic.Timeout(timeout[1], connect=timeout[0]))
    try:
        return client.messages.create(**params).to_dict()
    except anthropic.APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        detail = body.get('error') if isinstance(body.get('error'), dict) else {}
        kind = detail.get('type') or ''
        if 'credit balance' in str(detail.get('message', '')).lower():
            kind = 'credit_balance_exhausted'
        message = f'Claude HTTP {error.status_code} {kind}'.strip()
        raise (NotBilled if 400 <= error.status_code < 500 else RuntimeError)(message) from None
    except anthropic.APIConnectionError as error:  # Includes timeouts.
        raise RuntimeError('Claude connection failed: '+type(error).__name__) from None


def call_claude(request, timeout=(15, 600)):
    if not os.environ.get(ANTHROPIC_KEY):
        raise NotBilled(ANTHROPIC_KEY+' is missing')
    return from_claude(_send(claude_params(request), timeout), request)
