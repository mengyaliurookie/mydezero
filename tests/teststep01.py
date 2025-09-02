import unittest
from steps.step01 import Variable
import numpy as np

class TestStep01(unittest.TestCase):

    def test_variable(self):
        data=np.array(7)
        v=Variable(data)
        self.assertEqual(v.data,7)

if __name__=='__main__':
    unittest.main()