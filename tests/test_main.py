import tempfile
import unittest
from pathlib import Path

from main import AnalysisRequest, analyze_dataset, build_numeric_summary, is_dataset_question


class AnalysisTests(unittest.TestCase):
    def test_rejects_prompt_unrelated_to_dataset(self):
        request = AnalysisRequest(
            file_path="/tmp/not-used.csv",
            schema_json={"columns": [{"name": "amount", "type": "int64"}]},
            prompt="When is Christmas day?",
        )

        self.assertFalse(is_dataset_question(request.prompt, request.schema_json))

    def test_accepts_prompt_referencing_dataset_column(self):
        schema = {"columns": [{"name": "total_sales", "type": "int64"}]}

        self.assertTrue(is_dataset_question("What is total_sales?", schema))

    def test_builds_numeric_summary(self):
        import pandas as pd

        dataframe = pd.DataFrame({"amount": [10, 20, None]})

        self.assertEqual(
            build_numeric_summary(dataframe)["amount"],
            {
                "count": 2,
                "null_count": 1,
                "min": 10.0,
                "max": 20.0,
                "mean": 15.0,
                "sum": 30.0,
            },
        )

    def test_analyze_returns_full_dataset_metrics_and_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "sales.csv"
            csv_path.write_text("product,amount\nA,10\nB,20\nC,30\n")

            response = analyze_dataset(AnalysisRequest(
                file_path=str(csv_path),
                schema_json={"columns": []},
                prompt="Summarize the dataset and amount totals",
            ))

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["row_count"], 3)
        self.assertEqual(response["numeric_summary"]["amount"]["sum"], 60.0)
        self.assertEqual(len(response["result_sample"]), 3)
        self.assertEqual(response["result_sample"][0]["product"], "A")


if __name__ == "__main__":
    unittest.main()
