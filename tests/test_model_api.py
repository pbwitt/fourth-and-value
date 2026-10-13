"""Claude requests keep the bounded request shape and fail closed like OpenAI ones."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import model_api
from nhl.v2 import astra

SEARCH = dict(model='claude-opus-5-5', service_tier='default', store=False, reasoning={'effort': 'high'},
              max_output_tokens=8000, max_tool_calls=1, tool_choice='required',
              tools=[dict(type='web_search', search_context_size='low', filters={'allowed_domains': ['nfl.com', 'espn.com']})],
              instructions='Research.', input='{"games":[]}',
              text={'format': dict(type='json_schema', name='s', strict=True, schema=astra.obj({'directions': dict(
                  type='array', maxItems=3, items=astra.obj({'url': astra.string(100)}))}))})
REVIEW = dict(model='claude-opus-5-5', service_tier='default', store=False, reasoning={'effort': 'high'},
              max_output_tokens=12000, instructions='Review.', input='{}',
              text={'format': dict(type='json_schema', name='r', strict=True, schema=astra.SCHEMA)})


def message(content, stop='end_turn', **usage):
    return dict(id='msg_1', type='message', role='assistant', model='claude-opus-5-5', content=content,
                stop_reason=stop, usage=dict(dict(input_tokens=100, output_tokens=40), **usage))


class RequestTests(unittest.TestCase):
    def test_review_uses_json_outputs_without_unenforceable_limits(self):
        p = model_api.claude_params(REVIEW)
        self.assertEqual((p['model'], p['max_tokens'], p['system']), ('claude-opus-5-5', 12000, 'Review.'))
        self.assertEqual(p['messages'], [{'role': 'user', 'content': '{}'}])
        self.assertEqual(p['output_config']['effort'], 'high')
        schema = json.dumps(p['output_config']['format'])
        for word in ('maxLength', 'minLength', 'maxItems'):
            self.assertNotIn(word, schema)
        self.assertIn('"minItems": 1', schema)
        self.assertIn('needs_information', schema)
        # Claude cannot enforce the limits, so each one is stated where the model reads it.
        sent = p['output_config']['format']['schema']['properties']['reviews']
        assessment = sent['items']['properties']['assessment']['properties']
        self.assertEqual(assessment['reason']['description'], 'At most 350 characters.')
        self.assertEqual(sent['description'], 'At most 4 items.')
        self.assertNotIn('description', assessment['verdict'])
        self.assertNotIn('tools', p)
        # The original schema, with every limit, is still what validate_shape enforces.
        self.assertIn('maxLength', json.dumps(REVIEW['text']))

    def test_search_is_one_basic_domain_limited_search_with_prompted_json(self):
        p = model_api.claude_params(SEARCH)
        self.assertEqual(p['tools'], [dict(type='web_search_20250305', name='web_search', max_uses=1,
                                           allowed_domains=['nfl.com', 'espn.com'])])
        self.assertNotIn('format', p['output_config'])
        self.assertNotIn('tool_choice', p)
        self.assertIn('Run the web search', p['system'])
        self.assertIn('"maxItems":3', p['system'])
        with self.assertRaises(ValueError):
            model_api.claude_params(dict(SEARCH, tools=[{'type': 'file_search'}]))

    def test_effort_names_map_to_claude_levels(self):
        for given, expected in [('minimal', 'low'), ('low', 'low'), ('high', 'high'), (None, 'low')]:
            request = dict(REVIEW, reasoning={'effort': given} if given else {})
            self.assertEqual(model_api.claude_params(request)['output_config']['effort'], expected)


class ResponseTests(unittest.TestCase):
    def test_review_reply_matches_the_existing_response_fields(self):
        r = model_api.from_claude(message([dict(type='thinking', thinking='', signature='opaque'),
            dict(type='text', text='{"reviews":[]}', citations=None)], cache_read_input_tokens=7,
            cache_creation_input_tokens=3), REVIEW)
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['output'][-1]['content'][0]['text'], '{"reviews":[]}')
        self.assertEqual((r['usage']['input_tokens'], r['usage']['output_tokens']), (110, 40))
        self.assertNotIn('signature', json.dumps(r['anthropic_message']))

    def test_refusal_and_truncation_fail_closed(self):
        refused = model_api.from_claude(dict(message([], stop='refusal'), stop_details=dict(category='cyber')), REVIEW)
        self.assertEqual(refused['status'], 'incomplete')
        self.assertEqual(refused['output'][-1]['content'][0]['type'], 'refusal')
        with self.assertRaises(ValueError):
            astra.parse_response(refused, dict(candidates=[]), [], None)
        cut = model_api.from_claude(message([dict(type='text', text='{"rev')], stop='max_tokens'), REVIEW)
        self.assertEqual((cut['status'], cut['incomplete_details']['reason']), ('incomplete', 'max_tokens'))

    def test_search_sources_citations_and_final_json_only(self):
        content = [dict(type='text', text='Searching first.'),
                   dict(type='server_tool_use', id='s1', name='web_search', input={'query': 'q'}),
                   dict(type='web_search_tool_result', tool_use_id='s1', content=[
                       dict(type='web_search_result', url='https://www.nfl.com/news/a', title='A', encrypted_content='zz')]),
                   dict(type='text', text='```json\n{"directions":[{"url":"https://www.nfl.com/news/a"}', citations=[
                       dict(type='web_search_result_location', url='https://www.nfl.com/news/a', cited_text='x', encrypted_index='i')]),
                   dict(type='text', text=']}\n```')]
        r = model_api.from_claude(message(content, server_tool_use={'web_search_requests': 1}), SEARCH)
        self.assertEqual(r['usage']['web_search_requests'], 1)
        sources = [s['url'] for o in r['output'] if o['type'] == 'web_search_call' for s in o['action']['sources']]
        self.assertEqual(sources, ['https://www.nfl.com/news/a'])
        text = r['output'][-1]['content'][0]
        self.assertEqual(json.loads(text['text']), {'directions': [{'url': 'https://www.nfl.com/news/a'}]})
        self.assertEqual(text['annotations'][0]['type'], 'url_citation')
        self.assertNotIn('encrypted', json.dumps(r['anthropic_message']))

    def test_search_error_contributes_no_sources(self):
        content = [dict(type='web_search_tool_result', tool_use_id='s1',
                        content=dict(type='web_search_tool_result_error', error_code='unavailable')),
                   dict(type='text', text='{"directions":[]}')]
        r = model_api.from_claude(message(content), SEARCH)
        self.assertFalse([o for o in r['output'] if o['type'] == 'web_search_call'])


class RoutingTests(unittest.TestCase):
    def test_missing_key_never_sends(self):
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': ''}), patch.object(model_api, '_send') as send:
            with self.assertRaises(model_api.NotBilled):
                astra.call_api(REVIEW)
            send.assert_not_called()

    def test_claude_and_openai_names_choose_their_provider(self):
        reply = message([dict(type='text', text='{"reviews":[]}')])
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'synthetic'}), \
             patch.object(model_api, '_send', return_value=reply) as send, \
             patch.object(astra.requests, 'post') as post:
            self.assertEqual(astra.call_api(REVIEW)['provider'], 'anthropic')
            send.assert_called_once()
            self.assertEqual(send.call_args.args[1], (15, 300))
            post.assert_not_called()
        self.assertEqual((model_api.key_name('claude-sonnet-5-5'), model_api.key_name('gpt-6.1-sol')),
                         ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY'))
        # Switching back is only a model name: the OpenAI request is unchanged.
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'synthetic'}), patch.object(model_api, '_send') as send, \
             patch.object(astra.requests, 'post') as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {'status': 'completed'}
            astra.call_api(dict(REVIEW, model='gpt-6.1-sol'))
            self.assertEqual(post.call_args.kwargs['json']['model'], 'gpt-6.1-sol')
            send.assert_not_called()

    def test_api_rejections_are_unbilled_and_server_errors_stay_uncertain(self):
        import anthropic
        import httpx2

        def failure(code, kind, message='x'):
            response = httpx2.Response(code, request=httpx2.Request('POST', 'https://api.anthropic.com/v1/messages'))
            return anthropic.APIStatusError(message, response=response, body={'error': {'type': kind, 'message': message}})
        cases = [(failure(401, 'authentication_error'), model_api.NotBilled, 'Claude HTTP 401 authentication_error'),
                 (failure(400, 'invalid_request_error', 'Your credit balance is too low'), model_api.NotBilled,
                  'Claude HTTP 400 credit_balance_exhausted'),
                 (failure(529, 'overloaded_error'), RuntimeError, 'Claude HTTP 529 overloaded_error')]
        for raised, kind, text in cases:
            with patch.object(anthropic, 'Anthropic') as client:
                client.return_value.messages.create.side_effect = raised
                with self.assertRaises(RuntimeError) as caught:
                    model_api._send({}, (15, 300))
            self.assertIs(type(caught.exception), kind)
            self.assertEqual(str(caught.exception), text)

    def test_research_configs_name_the_budgeted_model(self):
        configs = [json.loads((astra.ROOT/'config'/name).read_text()) for name in ('analyst_review.json', 'nhl_analyst.json')]
        self.assertEqual({c['model'] for c in configs}, {astra.MODEL})
        self.assertIn(astra.MODEL, astra.RATES)


if __name__ == '__main__':
    unittest.main()
