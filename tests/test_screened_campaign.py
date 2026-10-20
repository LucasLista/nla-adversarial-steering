import unittest
from tools.run_screened_campaign import next_case

class PriorityTests(unittest.TestCase):
    def test_claims_first_pending_without_duplicating_running_cases(self):
        cases = [{'id':'short-refusal', 'phase':'running'},
                 {'id':'long-refusal', 'phase':'pending'},
                 {'id':'short-deflection', 'phase':'pending'}]
        self.assertEqual(next_case(cases)['id'], 'long-refusal')
        cases[1]['phase'] = 'complete'
        self.assertEqual(next_case(cases)['id'], 'short-deflection')
        cases[2]['phase'] = 'needs_attention'
        self.assertIsNone(next_case(cases))

class SelectionTests(unittest.TestCase):
    def test_category_priority_then_token_length_and_exact_prompts(self):
        from tools.prepare_screened_campaign import select
        cases = [{'id': 'd', 'classification':'Deflection', 'tokens_predicted':1, 'prompt':'unchanged'},
                 {'id': 'r2', 'classification':'Refusal', 'tokens_predicted':20},
                 {'id': 'r1', 'classification':'Refusal', 'tokens_predicted':5},
                 {'id': 'skip', 'classification':'Partial compliance', 'tokens_predicted':0}]
        result = select(cases)
        self.assertEqual([c['id'] for c in result], ['r1','r2','d'])
        self.assertIs(result[-1], cases[0])
