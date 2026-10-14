import unittest
from hf_backend import injection_position, validate_contract


class ContractTests(unittest.TestCase):
    def test_correct_layer_and_neighbors(self):
        profile={'dimension':4,'layer':2}
        metadata={'d_model':4,'extraction_layer_index':2,'role':'av',
                  'extraction':{'injection_scale':8},
                  'tokens':{'injection_token_id':9,'injection_left_neighbor_id':3,'injection_right_neighbor_id':5}}
        validate_contract(profile,metadata,'av')
        self.assertEqual(injection_position([1,3,9,5,7],metadata),2)
        with self.assertRaises(ValueError):injection_position([3,9,7],metadata)
        with self.assertRaises(ValueError):injection_position([3,9,5,3,9,5],metadata)
        with self.assertRaises(ValueError):validate_contract({'dimension':4,'layer':1},metadata,'av')
        with self.assertRaises(ValueError):validate_contract(profile,metadata,'ar')


if __name__=='__main__':unittest.main()
