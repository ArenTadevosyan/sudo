from __future__ import annotations

from .brain import Brain


def main() -> None:
    brain = Brain()
    print("AI Brain запущен. Команды: /learn, /teach, /goal, /fact, /reflect, /recall, /feedback, /exit")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/reflect":
            print(brain.reflect())
            continue
        if user_text == "/learn":
            print(brain.learn_report())
            continue
        if user_text.startswith("/goal "):
            print(brain.set_goal(user_text.removeprefix("/goal ")))
            continue
        if user_text.startswith("/fact "):
            print(brain.store_fact(user_text.removeprefix("/fact ")))
            continue
        if user_text.startswith("/teach "):
            payload = user_text.removeprefix("/teach ")
            question, separator, answer = payload.partition("=>")
            if not separator:
                print("Формат: /teach вопрос => хороший ответ")
            else:
                print(brain.teach(question, answer))
            continue
        if user_text.startswith("/recall "):
            query = user_text.removeprefix("/recall ")
            for item in brain.recall(query, limit=8):
                print(f"#{item.memory_id} {item.kind} score={item.score} strength={item.strength:.2f}: {item.text}")
            continue
        if user_text.startswith("/feedback "):
            _, _, payload = user_text.partition(" ")
            rating, _, note = payload.partition(" ")
            print(brain.feedback(rating, note))
            continue

        print(brain.think(user_text))


if __name__ == "__main__":
    main()
