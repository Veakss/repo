import unittest
from extractor import extract_markers

class TestMarkerExtractor(unittest.TestCase):
    def test_basic_extraction(self):
        text = "Hello {{name}}, your ID is {{user_id}}."
        expected = ["name", "user_id"]
        self.assertEqual(extract_markers(text), expected)

    def test_no_markers(self):
        text = "This text has no markers."
        expected = []
        self.assertEqual(extract_markers(text), expected)

    def test_empty_marker(self):
        text = "An empty marker: {{}}."
        expected = [""]
        self.assertEqual(extract_markers(text), expected)

if __name__ == '__main__':
    unittest.main()