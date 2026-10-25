import unittest
from datetime import date

from refund_timeline import calculate_refund_window, format_refund_window
from retriever import RetrievalHit


def make_hit(section_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(
        evidence_id=section_id,
        text=text,
        source_text=text,
        metadata={"section_id": section_id, "status": "current"},
        score=1.0,
    )


class RefundTimelineTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            make_hit(
                "REF-2",
                "| UPI | 1 to 3 business days |\n"
                "| Credit or debit card | 5 to 7 business days |\n"
                "| Net banking | 3 to 5 business days |\n"
                "| ZentroMart Wallet | Instant, within 2 hours |",
            ),
            make_hit(
                "KB-00.2",
                "All records are frozen on 20 October 2026. "
                'Treat this date as "today".',
            ),
            make_hit(
                "KB-00.3",
                "A business day is Monday to Saturday, excluding "
                "warehouse holidays.\n"
                "| 2026-10-19 | Warehouse closure |\n"
                "| 2026-10-02 | Gandhi Jayanti |",
            ),
        ]

    def test_excludes_initiation_day_and_skips_sunday_and_holiday(self):
        window = calculate_refund_window(
            "UPI refund started on 17 October 2026.",
            self.evidence,
        )

        self.assertIsNotNone(window)
        self.assertEqual(window.initiation_date, date(2026, 10, 17))
        self.assertEqual(window.earliest_date, date(2026, 10, 20))
        self.assertEqual(window.latest_date, date(2026, 10, 22))

    def test_credit_card_window_crosses_holiday_and_sunday(self):
        window = calculate_refund_window(
            "Credit card refund started on 12 October 2026.",
            self.evidence,
        )

        self.assertEqual(window.earliest_date, date(2026, 10, 17))
        self.assertEqual(window.latest_date, date(2026, 10, 21))

    def test_hindi_language_answer_keeps_policy_citations(self):
        window = calculate_refund_window(
            "Net banking refund started on 17 October 2026.",
            self.evidence,
        )

        answer = format_refund_window(window, "hi")

        self.assertIn("22 October 2026", answer)
        self.assertIn("24 October 2026", answer)
        self.assertIn("[REF-2] [KB-00.3]", answer)

    def test_does_not_calculate_without_both_current_policy_sections(self):
        incomplete = [
            self.evidence[0],
            self.evidence[1],
        ]

        self.assertIsNone(
            calculate_refund_window(
                "UPI refund started on 17 October 2026.",
                incomplete,
            )
        )

    def test_does_not_infer_a_business_day_range_for_wallet(self):
        self.assertIsNone(
            calculate_refund_window(
                "ZentroMart Wallet refund started on 17 October 2026.",
                self.evidence,
            )
        )


if __name__ == "__main__":
    unittest.main()
