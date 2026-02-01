import unittest
from unittest.mock import MagicMock, patch

from pipeline.blog_workflow import AEOBlogPipeline


class TestAEOPipeline(unittest.TestCase):
    @patch.object(AEOBlogPipeline, "_get_completion")
    @patch.object(AEOBlogPipeline, "_search_web", return_value="Research summary")
    @patch.object(AEOBlogPipeline, "_retrieve_knowledge", return_value="Guidelines")
    def test_pipeline_flow(self, mock_retrieve, mock_search, mock_completion):
        mock_completion.side_effect = [
            "Outline",
            "Draft content",
            "Optimization report",
            "Final blog post",
        ]

        fake_model = MagicMock()
        fake_vector_db = MagicMock()

        pipeline = AEOBlogPipeline(genai_client=fake_model, vector_db=fake_vector_db)
        result = pipeline.run("Test Topic")

        self.assertEqual(mock_completion.call_count, 4)
        mock_search.assert_called_once()
        mock_retrieve.assert_called_once_with("AEO guidelines content structure")
        self.assertIn("Final blog post", result)


if __name__ == "__main__":
    unittest.main()
