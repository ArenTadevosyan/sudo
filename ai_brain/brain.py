from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = ROOT / "rust" / "brain_core" / "Cargo.toml"
DEFAULT_MEMORY = ROOT / "data" / "memory.tsv"


@dataclass
class Recall:
    score: int
    memory_id: int
    kind: str
    importance: float
    strength: float
    uses: int
    text: str
    tags: str


class Brain:
    """Python layer: goals, dialogue policy, and self-improvement loop."""

    def __init__(self, memory_path: Path | str = DEFAULT_MEMORY) -> None:
        self.memory_path = Path(memory_path)
        self.identity = (
            "Я локальный ИИ-мозг: Python думает и ведет диалог, "
            "Rust быстро хранит, усиливает и ищет память."
        )

    def remember(
        self,
        text: str,
        tags: str = "dialogue",
        importance: float | None = None,
        kind: str = "dialogue",
    ) -> int:
        args = ["remember", str(self.memory_path), kind, text, tags]
        if importance is not None:
            args.append(f"{importance:.3f}")
        output = self._core(args)
        return int(output.strip())

    def recall(self, query: str, limit: int = 5, kind: str = "any") -> list[Recall]:
        output = self._core(["search", str(self.memory_path), query, str(limit), kind])
        recalls: list[Recall] = []
        for line in output.splitlines():
            parts = line.split("\t", 7)
            if len(parts) == 8:
                recalls.append(
                    Recall(
                        score=int(parts[0]),
                        memory_id=int(parts[1]),
                        kind=self._unescape(parts[2]),
                        importance=float(parts[3]),
                        strength=float(parts[4]),
                        uses=int(parts[5]),
                        text=self._unescape(parts[6]),
                        tags=self._unescape(parts[7]),
                    )
                )
                continue
            if len(parts) == 5:
                recalls.append(
                    Recall(
                        score=int(parts[0]),
                        memory_id=int(parts[1]),
                        kind="dialogue",
                        importance=float(parts[2]),
                        strength=0.5,
                        uses=0,
                        text=self._unescape(parts[3]),
                        tags=self._unescape(parts[4]),
                    )
                )
        return recalls

    def reflect(self) -> str:
        return self._core(["reflect", str(self.memory_path)]).strip()

    def learn_report(self) -> str:
        rules = self.recall("правило развитие feedback ошибка цель", limit=5, kind="rule")
        goals = self.recall("цель план результат", limit=5, kind="goal")
        lines = [self.reflect(), "", "Активные правила:"]
        lines.extend(self._format_recall(item) for item in rules)
        lines.append("Активные цели:")
        lines.extend(self._format_recall(item) for item in goals)
        return "\n".join(line for line in lines if line)

    def set_goal(self, goal: str) -> str:
        goal = goal.strip()
        if not goal:
            return "Цель пустая. Напиши: /goal чему учиться."
        self.remember(f"Цель: {goal}", "goal,active", importance=0.9, kind="goal")
        self.remember(
            "Правило развития: для активной цели нужно предлагать следующий проверяемый шаг.",
            "principle,evolution,goal",
            importance=0.82,
            kind="rule",
        )
        return f"Цель сохранена: {goal}"

    def teach(self, question: str, answer: str) -> str:
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return "Формат: /teach вопрос => хороший ответ."
        self.remember(f"Пользователь: {question}", "teach,user", importance=0.86, kind="dialogue")
        self.remember(f"ИИ: {answer}", "teach,assistant", importance=0.9, kind="dialogue")
        self.remember(
            "Правило развития: ответы из /teach считать эталонными примерами диалога.",
            "principle,evolution,teach",
            importance=0.88,
            kind="rule",
        )
        return "Пример обучения сохранен."

    def store_fact(self, fact: str) -> str:
        fact = fact.strip()
        if not fact:
            return "Факт пустой. Напиши: /fact что запомнить."
        self.remember(f"Факт: {fact}", "fact,user", importance=0.78, kind="fact")
        return "Факт сохранен."

    def feedback(self, rating: str, note: str = "") -> str:
        rating = rating.strip().lower()
        note = note.strip()
        if rating in {"+", "good", "хорошо", "да"}:
            self.remember(
                f"Положительная обратная связь. {note}",
                "feedback,positive",
                importance=0.85,
                kind="feedback",
            )
            self._reinforce(note or "положительная обратная связь", 0.09)
            return "Принял: усиливаю похожий стиль ответа."
        if rating in {"-", "bad", "плохо", "нет"}:
            self.remember(
                f"Отрицательная обратная связь. {note}",
                "feedback,negative",
                importance=0.95,
                kind="feedback",
            )
            self.remember(
                "Правило развития: после отрицательной обратной связи нужно отвечать короче и задавать уточнение.",
                "principle,evolution,feedback",
                importance=0.9,
                kind="rule",
            )
            self._reinforce("отрицательная обратная связь короче уточнение", 0.12)
            return "Принял: отмечаю ошибку, усиливаю правило исправления."
        return "Оценка не распознана. Используй + или -."

    def think(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return "Я слушаю. Дай мне мысль, цель или задачу."

        memories = self.recall(user_text)
        rules = self.recall(user_text, limit=3, kind="rule")
        goals = self.recall("активная цель план шаг", limit=2, kind="goal")
        context = self._context_sentence(memories)
        rule_hint = self._rule_sentence(rules)
        goal_hint = self._goal_sentence(goals)

        lowered = user_text.lower()
        if any(word in lowered for word in ["кто ты", "что ты", "identity"]):
            answer = self.identity
        elif any(word in lowered for word in ["план", "сделай", "создай", "развивай"]):
            answer = (
                f"{context} {rule_hint}{goal_hint}"
                "План: 1. сохранить цель, 2. найти похожий опыт, "
                "3. выполнить маленький шаг, 4. записать результат и feedback."
            )
        elif lowered.endswith("?") or any(word in lowered for word in ["почему", "как", "зачем"]):
            answer = (
                f"{context} {rule_hint}"
                "Мой ответ основан на памяти, правилах и обратной связи. "
                "После оценки я усилю удачные правила или запишу исправление."
            )
        else:
            answer = (
                f"{context} {goal_hint}"
                "Я запомнил это и свяжу с будущими запросами. "
                "Полезный следующий шаг: /feedback + или /feedback - с причиной."
            )

        self.remember(f"Пользователь: {user_text}", "user,input", kind="dialogue")
        self.remember(f"ИИ: {answer}", "assistant,output", importance=0.45, kind="dialogue")
        self.evolve(user_text, answer)
        self._reinforce(user_text, 0.03)
        return answer

    def evolve(self, user_text: str, answer: str) -> None:
        signal = user_text.lower()
        if "запомни" in signal or "важно" in signal:
            self.remember(
                "Правило развития: сообщения с явным маркером важности получают больший вес.",
                "principle,evolution,importance",
                importance=0.9,
                kind="rule",
            )
        if "цель" in signal or "науч" in signal:
            self.remember(
                f"Цель-кандидат из диалога: {user_text}",
                "goal,candidate",
                importance=0.72,
                kind="goal",
            )
        if len(answer) > 220:
            self.remember(
                "Правило развития: длинные ответы нужно сжимать, если пользователь не просил детали.",
                "principle,evolution,brevity",
                importance=0.7,
                kind="rule",
            )

    def _reinforce(self, query: str, delta: float) -> int:
        if not query.strip():
            return 0
        output = self._core(["reinforce", str(self.memory_path), query, f"{delta:.3f}"])
        return int(output.strip())

    def _context_sentence(self, memories: list[Recall]) -> str:
        if not memories:
            return "В памяти пока нет близкого опыта."

        best = memories[0].text
        if len(best) > 140:
            best = best[:137].rstrip() + "..."
        return f"Похожее воспоминание: {best}"

    def _rule_sentence(self, rules: list[Recall]) -> str:
        if not rules:
            return ""
        return f"Правило: {rules[0].text} "

    def _goal_sentence(self, goals: list[Recall]) -> str:
        if not goals:
            return ""
        return f"Активная цель: {goals[0].text}. "

    def _format_recall(self, item: Recall) -> str:
        return f"#{item.memory_id} {item.kind} strength={item.strength:.2f}: {item.text}"

    def _core(self, args: list[str]) -> str:
        command = [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(CORE_MANIFEST),
            "--",
            *args,
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "brain_core failed")
        return completed.stdout

    @staticmethod
    def _unescape(value: str) -> str:
        result = []
        escaping = False
        for char in value:
            if not escaping and char == "\\":
                escaping = True
                continue
            if escaping:
                result.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(char, char))
                escaping = False
            else:
                result.append(char)
        if escaping:
            result.append("\\")
        return "".join(result)
