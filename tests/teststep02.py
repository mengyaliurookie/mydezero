from steps.step02 import Square
from steps.step01 import Variable
import unittest
import numpy as np

class TestStep02(unittest.TestCase):
    def setUp(self):
        self.x = Variable(np.array(10))
        self.f=Square()
    def test_forward(self):
        y = self.f(self.x)
        expected = np.array(100)
        self.assertEqual(y.data,expected)