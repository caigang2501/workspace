import unittest

def add(a, b):
    return a + b

class TestAddFunction(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(3, 5), 8)
    
    def test_add_negative_numbers(self):
        self.assertEqual(add(-3, -5), -8)
    
    def test_add_mixed_numbers(self):
        self.assertEqual(add(10, -5), 5)

if __name__ == '__main__':
    unittest.main()
