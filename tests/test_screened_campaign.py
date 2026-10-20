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
