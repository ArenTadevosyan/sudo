use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct Memory {
    id: usize,
    timestamp_ms: u128,
    kind: String,
    importance: f32,
    strength: f32,
    uses: u32,
    text: String,
    tags: Vec<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("brain_core: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("help");

    match command {
        "remember" => {
            let path = need(&args, 2, "memory path")?;
            let kind = args.get(3).map(String::as_str).unwrap_or("dialogue");
            let text = need(&args, 4, "text")?;
            let tags = args.get(5).map(String::as_str).unwrap_or("");
            let importance = args
                .get(6)
                .and_then(|raw| raw.parse::<f32>().ok())
                .unwrap_or_else(|| estimate_importance(kind, text));
            let id = remember(path, kind, text, tags, importance)?;
            println!("{id}");
        }
        "search" => {
            let path = need(&args, 2, "memory path")?;
            let query = need(&args, 3, "query")?;
            let limit = args
                .get(4)
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(5);
            let kind_filter = args.get(5).map(String::as_str).unwrap_or("any");
            for (score, memory) in search(path, query, limit, kind_filter)? {
                println!(
                    "{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}",
                    score,
                    memory.id,
                    escape(&memory.kind),
                    memory.importance,
                    memory.strength,
                    memory.uses,
                    escape(&memory.text),
                    escape(&memory.tags.join(","))
                );
            }
        }
        "reinforce" => {
            let path = need(&args, 2, "memory path")?;
            let query = need(&args, 3, "query")?;
            let delta = args
                .get(4)
                .and_then(|raw| raw.parse::<f32>().ok())
                .unwrap_or(0.08);
            let changed = reinforce(path, query, delta)?;
            println!("{changed}");
        }
        "reflect" => {
            let path = need(&args, 2, "memory path")?;
            for line in reflect(path)? {
                println!("{line}");
            }
        }
        _ => print_help(),
    }

    Ok(())
}

fn print_help() {
    println!("brain_core commands:");
    println!("  remember  <memory-file> <kind> <text> [tags] [importance]");
    println!("  search    <memory-file> <query> [limit] [kind|any]");
    println!("  reinforce <memory-file> <query> [delta]");
    println!("  reflect   <memory-file>");
}

fn need<'a>(args: &'a [String], index: usize, name: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("missing {name}"))
}

fn remember(
    path: &str,
    kind: &str,
    text: &str,
    tags: &str,
    importance: f32,
) -> Result<usize, String> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let next_id = load(path)?.len() + 1;
    let timestamp_ms = now_ms()?;
    let strength = base_strength(kind, importance);
    let record = format!(
        "{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\n",
        next_id,
        timestamp_ms,
        escape(kind),
        importance.clamp(0.0, 1.0),
        strength,
        0,
        escape(text),
        escape(tags)
    );

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| err.to_string())?;
    file.write_all(record.as_bytes())
        .map_err(|err| err.to_string())?;
    Ok(next_id)
}

fn search(
    path: &str,
    query: &str,
    limit: usize,
    kind_filter: &str,
) -> Result<Vec<(u32, Memory)>, String> {
    let query_terms = tokenize(query);
    let mut scored = Vec::new();

    for memory in load(path)? {
        if kind_filter != "any" && memory.kind != kind_filter {
            continue;
        }
        let memory_terms = tokenize(&memory.text);
        let overlap = query_terms.intersection(&memory_terms).count() as u32;
        let tag_bonus = memory
            .tags
            .iter()
            .flat_map(|tag| tokenize(tag))
            .filter(|tag| query_terms.contains(tag))
            .count() as u32;
        let kind_bonus = if query_terms.contains(&stem(&memory.kind)) {
            5
        } else {
            0
        };
        let importance_bonus = (memory.importance * 4.0).round() as u32;
        let strength_bonus = (memory.strength * 5.0).round() as u32;
        let score = overlap * 10 + tag_bonus * 4 + kind_bonus + importance_bonus + strength_bonus;

        if score > 0 {
            scored.push((score, memory));
        }
    }

    scored.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.strength.total_cmp(&left.1.strength))
            .then_with(|| right.1.timestamp_ms.cmp(&left.1.timestamp_ms))
    });
    scored.truncate(limit);
    Ok(scored)
}

fn reinforce(path: &str, query: &str, delta: f32) -> Result<usize, String> {
    let query_terms = tokenize(query);
    let mut memories = load(path)?;
    let mut changed = 0;

    for memory in &mut memories {
        let memory_terms = tokenize(&memory.text);
        let overlap = query_terms.intersection(&memory_terms).count();
        if overlap > 0 {
            memory.strength = (memory.strength + delta + overlap as f32 * 0.015).clamp(0.0, 1.0);
            memory.uses += 1;
            changed += 1;
        }
    }

    save_all(path, &memories)?;
    Ok(changed)
}

fn reflect(path: &str) -> Result<Vec<String>, String> {
    let memories = load(path)?;
    if memories.is_empty() {
        return Ok(vec!["Память пока пустая.".to_string()]);
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut kinds: HashMap<String, usize> = HashMap::new();
    let mut rules = 0;
    let mut goals = 0;
    let mut feedback = 0;
    let mut total_strength = 0.0;

    for memory in &memories {
        *kinds.entry(memory.kind.clone()).or_insert(0) += 1;
        total_strength += memory.strength;
        if memory.kind == "rule" {
            rules += 1;
        } else if memory.kind == "goal" {
            goals += 1;
        } else if memory.kind == "feedback" {
            feedback += 1;
        }
        for token in tokenize(&memory.text) {
            if token.len() > 3 {
                *counts.entry(token).or_insert(0) += 1;
            }
        }
    }

    let mut terms: Vec<(String, usize)> = counts.into_iter().collect();
    terms.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    terms.truncate(8);

    let mut kind_list: Vec<(String, usize)> = kinds.into_iter().collect();
    kind_list.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let focus = terms
        .iter()
        .map(|(term, count)| format!("{term}:{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let kind_text = kind_list
        .iter()
        .map(|(kind, count)| format!("{kind}:{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let avg_strength = total_strength / memories.len() as f32;

    Ok(vec![
        format!("Всего воспоминаний: {}", memories.len()),
        format!("Типы памяти: {kind_text}"),
        format!("Средняя сила памяти: {avg_strength:.2}"),
        format!("Цели: {goals}, правила: {rules}, обратная связь: {feedback}"),
        format!("Частые темы: {focus}"),
        "Следующий шаг: учиться на feedback, усиливать правила, закрывать цели результатами."
            .to_string(),
    ])
}

fn load(path: &str) -> Result<Vec<Memory>, String> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.to_string()),
    };

    let mut memories = Vec::new();
    for line in raw.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() == 8 {
            memories.push(parse_v2(&parts, memories.len() + 1));
        } else if parts.len() == 5 {
            memories.push(parse_v1(&parts, memories.len() + 1));
        }
    }

    Ok(memories)
}

fn parse_v2(parts: &[&str], fallback_id: usize) -> Memory {
    let kind = unescape(parts[2]);
    let importance = parts[3].parse::<f32>().unwrap_or(0.5);
    Memory {
        id: parts[0].parse::<usize>().unwrap_or(fallback_id),
        timestamp_ms: parts[1].parse::<u128>().unwrap_or(0),
        kind,
        importance,
        strength: parts[4].parse::<f32>().unwrap_or(0.5),
        uses: parts[5].parse::<u32>().unwrap_or(0),
        text: unescape(parts[6]),
        tags: split_tags(parts[7]),
    }
}

fn parse_v1(parts: &[&str], fallback_id: usize) -> Memory {
    let text = unescape(parts[3]);
    let tags = split_tags(parts[4]);
    let kind = infer_kind(&tags, &text);
    let importance = parts[2].parse::<f32>().unwrap_or(0.5);
    Memory {
        id: parts[0].parse::<usize>().unwrap_or(fallback_id),
        timestamp_ms: parts[1].parse::<u128>().unwrap_or(0),
        kind,
        importance,
        strength: base_strength("dialogue", importance),
        uses: 0,
        text,
        tags,
    }
}

fn split_tags(raw: &str) -> Vec<String> {
    unescape(raw)
        .split(',')
        .filter(|tag| !tag.trim().is_empty())
        .map(|tag| tag.trim().to_string())
        .collect()
}

fn infer_kind(tags: &[String], text: &str) -> String {
    let lowered = text.to_lowercase();
    if tags
        .iter()
        .any(|tag| tag == "feedback" || tag == "positive" || tag == "negative")
    {
        "feedback".to_string()
    } else if tags
        .iter()
        .any(|tag| tag == "principle" || tag == "evolution")
    {
        "rule".to_string()
    } else if lowered.starts_with("цель:") || lowered.starts_with("цель-кандидат") {
        "goal".to_string()
    } else if lowered.starts_with("факт:") {
        "fact".to_string()
    } else {
        "dialogue".to_string()
    }
}

fn save_all(path: &str, memories: &[Memory]) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut output = String::new();
    for memory in memories {
        output.push_str(&format!(
            "{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\n",
            memory.id,
            memory.timestamp_ms,
            escape(&memory.kind),
            memory.importance.clamp(0.0, 1.0),
            memory.strength.clamp(0.0, 1.0),
            memory.uses,
            escape(&memory.text),
            escape(&memory.tags.join(","))
        ));
    }
    fs::write(path, output).map_err(|err| err.to_string())
}

fn tokenize(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|part| part.len() > 2)
        .map(stem)
        .collect()
}

fn stem(token: &str) -> String {
    for suffix in [
        "иями", "ями", "ами", "ого", "ему", "ыми", "ими", "ая", "ое", "ые", "ий", "ый", "ой", "ам",
        "ям", "ах", "ях", "ов", "ев", "ом", "ем", "и", "ы", "а", "я", "е", "у",
    ] {
        if token.len() > suffix.len() + 3 && token.ends_with(suffix) {
            return token.trim_end_matches(suffix).to_string();
        }
    }
    token.to_string()
}

fn estimate_importance(kind: &str, text: &str) -> f32 {
    let urgent = [
        "важно",
        "цель",
        "ошибка",
        "запомни",
        "нужно",
        "люблю",
        "ненавижу",
    ];
    let lowered = text.to_lowercase();
    let hits = urgent
        .iter()
        .filter(|word| lowered.contains(**word))
        .count() as f32;
    let kind_bonus = match kind {
        "rule" => 0.25,
        "goal" => 0.22,
        "feedback" => 0.18,
        "fact" => 0.12,
        _ => 0.0,
    };
    (0.32 + kind_bonus + hits * 0.15 + (text.len().min(240) as f32 / 240.0) * 0.22).clamp(0.1, 1.0)
}

fn base_strength(kind: &str, importance: f32) -> f32 {
    let kind_bonus = match kind {
        "rule" => 0.2,
        "goal" => 0.15,
        "feedback" => 0.12,
        "fact" => 0.08,
        _ => 0.0,
    };
    (0.25 + importance * 0.45 + kind_bonus).clamp(0.1, 1.0)
}

fn now_ms() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| err.to_string())
        .map(|duration| duration.as_millis())
}

fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn unescape(value: &str) -> String {
    let mut output = String::new();
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }

        match chars.next() {
            Some('t') => output.push('\t'),
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('\\') => output.push('\\'),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => output.push('\\'),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping_round_trip_keeps_record_shape() {
        let text = "строка\tс\nпереносом\\и слешем";
        assert_eq!(unescape(&escape(text)), text);
    }

    #[test]
    fn tokenizer_normalizes_simple_russian_suffixes() {
        let tokens = tokenize("Важные цели и важными задачами");
        assert!(tokens.contains("важн"));
        assert!(tokens.contains("цел"));
        assert!(tokens.contains("задач"));
    }

    #[test]
    fn importance_reacts_to_priority_words_and_kind() {
        assert!(
            estimate_importance("goal", "важно запомни цель")
                > estimate_importance("dialogue", "обычная фраза")
        );
    }

    #[test]
    fn old_memory_format_is_supported() {
        let goal_parts = ["1", "42", "0.600", "Цель: учиться", "user,input"];
        let goal = parse_v1(&goal_parts, 99);
        assert_eq!(goal.id, 1);
        assert_eq!(goal.kind, "goal");
        assert!(goal.strength > 0.0);

        let dialogue_parts = [
            "2",
            "43",
            "0.600",
            "ИИ: сохранить цель в плане",
            "assistant,output",
        ];
        let dialogue = parse_v1(&dialogue_parts, 99);
        assert_eq!(dialogue.kind, "dialogue");
    }
}
