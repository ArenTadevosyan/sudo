import tempfile
import unittest
from pathlib import Path

from ai_brain import Brain


class BrainTests(unittest.TestCase):
    def test_think_stores_and_recalls_typed_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "memory.tsv")
            answer = brain.think("Важно запомни: я развиваю локальный мозг")

            self.assertIn("запомнил", answer.lower() + brain.reflect().lower())
            recalls = brain.recall("локальный мозг")
            self.assertTrue(recalls)
            self.assertTrue(all(item.kind for item in recalls))

    def test_feedback_creates_rule_and_reinforces_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "memory.tsv")
            answer = brain.feedback("-", "слишком длинно")

            self.assertIn("ошибку", answer)
            recalls = brain.recall("отрицательной обратной связи", kind="rule")
            self.assertTrue(any("короче" in item.text for item in recalls))
            self.assertTrue(any(item.strength > 0.0 for item in recalls))

    def test_teach_creates_dialogue_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "memory.tsv")
            self.assertIn("сохранен", brain.teach("как тебя зовут?", "я локальный мозг"))

            recalls = brain.recall("как тебя зовут")
            self.assertTrue(any("Пользователь:" in item.text for item in recalls))
            self.assertTrue(any("ИИ:" in item.text for item in brain.recall("локальный мозг")))

    def test_goal_and_fact_are_first_class_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            brain = Brain(Path(directory) / "memory.tsv")
            self.assertIn("Цель сохранена", brain.set_goal("учиться на обратной связи"))
            self.assertIn("Факт сохранен", brain.store_fact("пользователь любит короткие ответы"))

            goals = brain.recall("обратной связи", kind="goal")
            facts = brain.recall("короткие ответы", kind="fact")
            self.assertTrue(goals)
            self.assertTrue(facts)
            self.assertIn("Активные цели", brain.learn_report())




class ImportDatasetTests(unittest.TestCase):
    def test_clean_text_trims_whitespace_and_length(self) -> None:
        from tools.import_hf_dataset import clean_text

        self.assertEqual(clean_text("  a\n b   c  ", 20), "a b c")
        self.assertEqual(clean_text("один два три", 9), "один два")

    def test_write_samples_counts_records(self) -> None:
        from tools.import_hf_dataset import write_samples

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.txt"
            count = write_samples(path, ["<TASK>dialogue\n<END>", "<TEXT>пример\n<END>"])

            self.assertEqual(count, 2)
            self.assertIn("<TEXT>пример", path.read_text())


if __name__ == "__main__":
    unittest.main()
