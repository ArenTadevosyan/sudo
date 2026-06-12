# AI Brain: Rust + Python

Локальный фундамент “мозга” для ИИ:

- Rust (`rust/brain_core`) хранит типизированную память, ищет похожие воспоминания, усиливает полезные записи и делает рефлексию.
- Python (`ai_brain`) управляет диалогом, целями, фактами, правилами, обратной связью и циклом обучения.
- Данные сохраняются в `data/memory.tsv`. Формат совместим со старой памятью и автоматически понимает новые поля.

## Запуск

```bash
python3 -m ai_brain.cli
```

Демо:

```bash
python3 demo.py
```

Проверка Rust-ядра:

```bash
cargo run --manifest-path rust/brain_core/Cargo.toml -- reflect data/memory.tsv
```

## Команды

- `/learn` — отчет обучения: состояние памяти, правила и цели.
- `/teach вопрос => ответ` — добавить эталонный диалог для обучения речи.
- `/goal текст` — сохранить активную цель обучения.
- `/fact текст` — сохранить устойчивый факт.
- `/reflect` — посмотреть состояние памяти и частые темы.
- `/recall текст` — найти похожие воспоминания.
- `/feedback + текст` или `/feedback - текст` — дать мозгу обратную связь и усилить правила.
- `/exit` — выйти.

## Цикл обучения

1. Пользователь задает цель через `/goal`.
2. Мозг сохраняет диалог как `dialogue`, факты как `fact`, цели как `goal`, правила как `rule`, оценки как `feedback`.
3. Перед ответом он ищет похожую память, активные цели и правила.
4. После ответа он записывает новый опыт.
5. После `/feedback` он усиливает похожие записи и создает правило исправления.

## Обучение языковой модели с нуля

Этот слой уже требует PyTorch и GPU. На локальной машине без GPU можно готовить датасет, а обучение запускать на RunPod или другом сервере.

Установка на GPU-сервере:

```bash
bash runpod_setup.sh
```

Если настраиваешь вручную:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/export_dataset.py
```

Запуск маленького Transformer с нуля:

```bash
python train_tiny_lm.py --device cuda --steps 3000
```

Для RTX 3090 24 GB можно начать так:

```bash
python train_tiny_lm.py \
  --device cuda \
  --batch-size 32 \
  --block-size 256 \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384 \
  --steps 5000
```

После обучения:

```bash
python generate_tiny_lm.py \
  --device cuda \
  --prompt $'<USER>как тебя зовут?\n<ASSISTANT>'
```

Что важно: текущий `data/memory.tsv` маленький, поэтому первая модель будет говорить слабо. Нужно наращивать данные через диалоги, `/teach`, `/goal`, `/fact`, `/feedback`, а также добавлять `.txt/.md` тексты в `dataset/raw/`.

## Импорт готовых датасетов

Для роста речи можно подтянуть маленький сэмпл с Hugging Face. Начинай с небольших лимитов, особенно на T4.

Диалоги OpenAssistant на русском:

```bash
pip install -r requirements.txt
python tools/import_hf_dataset.py --source oasst_ru --limit 5000
python tools/export_dataset.py
```

Русская Wikipedia для общей грамматики и знаний:

```bash
python tools/import_hf_dataset.py --source wiki_ru --limit 3000
python tools/export_dataset.py
```

Русский C4/web corpus лучше брать осторожно и только маленькими порциями:

```bash
python tools/import_hf_dataset.py --source c4_ru --limit 1000
python tools/export_dataset.py
```

После импорта можно обучать как обычно:

```bash
python train_tiny_lm.py --device cuda --batch-size 16 --block-size 128 --n-layer 4 --n-head 4 --n-embd 256 --steps 3000
```

Dialogue-only экспорт для обучения именно ответам, без служебной памяти и Wikipedia. Файл `dataset/raw/personality_ru.txt` автоматически усиливается, чтобы модель запомнила личность локального мозга:

```bash
python tools/export_dataset.py --mode dialogue --personality-repeat 30
python train_tiny_lm.py --device cuda --out checkpoints/dialogue_lm_v2 --batch-size 16 --block-size 128 --n-layer 4 --n-head 4 --n-embd 256 --steps 8000
```

Если модель говорит общими фразами, сделай короткий personality fine-tune поверх уже обученного dialogue checkpoint:

```bash
python tools/export_dataset.py --mode personality --personality-repeat 80
python train_tiny_lm.py \
  --device cuda \
  --init-from checkpoints/dialogue_lm_v2 \
  --out checkpoints/personality_lm \
  --batch-size 16 \
  --lr 1e-4 \
  --steps 1200
```

Проверка:

```bash
python generate_tiny_lm.py --device cuda --checkpoint-dir checkpoints/personality_lm --temperature 0.15 --top-k 4 --prompt $'<USER>как тебя зовут?\n<ASSISTANT>'
```

## Следующие сильные шаги

1. Добавить фильтр качества и дедупликацию для импортированных датасетов.
2. Подключить обученную модель к `Brain.think()` как генератор ответа.
3. Сделать планировщик: цель -> шаг -> проверка -> вывод -> новое правило.
4. Потом перейти от char-level модели к BPE-токенизатору.
5. Добавить векторный поиск по памяти.
