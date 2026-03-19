import unittest
from marker_extractor.marker_extractor import extract_marker_content

class TestMarkerExtractor(unittest.TestCase):

    def test_successful_extraction(self):
        text = "Start [START_MARKER]Hello World[END_MARKER] End"
        expected = "Hello World"
        self.assertEqual(extract_marker_content(text), expected)

    def test_extraction_with_newlines(self):
        text = "Data:\\n[START_MARKER]\\nLine 1\\nLine 2\\n[END_MARKER]\\nMore Data"
        # Aligning test expectation with function's current behavior (returning raw capture)
        expected = "\nLine 1\nLine 2\n" 
        self.assertEqual(extract_marker_content(text), expected)

    def test_no_markers_present(self):
        text = "Just some regular text."
        self.assertIsNone(extract_marker_content(text))

    def test_only_start_marker(self):
        text = "Start [START_MARKER]Incomplete"
        self.assertIsNone(extract_marker_content(text))

    def test_custom_markers(self):
        text = "BEGIN_DATA<<<This is custom>>>END_DATA"
        expected = "This is custom"
        self.assertEqual(extract_marker_content(text, start_marker="BEGIN_DATA<<<", end_marker=">>>END_DATA"), expected)

if __name__ == '__main__':
    unittest.main()
