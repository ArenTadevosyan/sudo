import argparse
import torch
import tiktoken
from pathlib import Path
from ai_brain.tiny_lm import build_model, load_config

def generate_code(prompt: str, checkpoint_dir: Path, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config_path = checkpoint_dir / "config.json"
    ckpt_path = checkpoint_dir / "model.pt"
    
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint or config not found in {checkpoint_dir}")
        
    config = load_config(config_path)
    model = build_model(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    input_ids = enc.encode_ordinary(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print(f"Generating on {device}...")
    with torch.no_grad():
        out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
    generated_text = enc.decode(out_idx[0].tolist())
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate code using trained agent.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to start code generation.")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/coder_agent"), help="Checkpoint directory")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    args = parser.parse_args()
    
    output = generate_code(args.prompt, args.ckpt, args.max_tokens, args.temperature)
    print("\n--- Generated Code ---\n")
    print(output)
    print("\n----------------------\n")
