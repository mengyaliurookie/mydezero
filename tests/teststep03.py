import unittest
import numpy as np
from steps.step02 import Exp,Square
from steps.step01 import Variable

class TestStep03(unittest.TestCase):
    def setUp(self):
        self.a=Square()
        self.b=Exp()
        self.c=Square()
        self.x = Variable(np.array(10))
        
    def test_forward(self):
        a=self.a(self.x)
        b=self.b(a)
        y=self.c(b)
        self.assertEqual(y.data,np.exp(100)**2)