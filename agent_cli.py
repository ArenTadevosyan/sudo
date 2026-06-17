import argparse
import torch
import tiktoken
from pathlib import Path

from ai_brain.tiny_lm import build_model, load_config
from ai_brain.brain import Brain

def load_agent_model(checkpoint_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = checkpoint_dir / "config.json"
    ckpt_path = checkpoint_dir / "model.pt"
    
    if not config_path.exists() or not ckpt_path.exists():
        print(f"⚠️  No trained model found at {checkpoint_dir}. Using Rust memory only mode.")
        return None, None, device
        
    config = load_config(config_path)
    model = build_model(config).to(device)
    
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    return model, enc, device

def main():
    parser = argparse.ArgumentParser(description="Powerful Coding Agent CLI")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/coder_agent"), help="Model checkpoint dir")
    args = parser.parse_args()
    
    print("🧠 Initializing Agent Memory (Rust Core)...")
    brain = Brain()
    
    print("⚙️  Loading Neural Network (PyTorch)...")
    model, enc, device = load_agent_model(args.ckpt)
    
    print("\n🚀 Agent is ready! Type your request, or type /quit to exit.")
    print("Commands:")
    print("  /goal <text>   - Set a long-term goal for the agent")
    print("  /rule <text>   - Add a rule to the agent's memory")
    print("  /feedback <+/-> <reason> - Give feedback on the last action")
    print("  <any prompt>   - Ask the agent to write code")
    
    while True:
        try:
            user_input = input("\n[User]> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/quit", "exit"]:
                break
                
            if user_input.startswith("/goal "):
                print("[Agent Memory]>", brain.set_goal(user_input[6:]))
                continue
                
            if user_input.startswith("/rule "):
                brain.remember(user_input[6:], tags="rule,user_defined", kind="rule", importance=0.9)
                print("[Agent Memory]> Rule saved to memory.")
                continue
                
            if user_input.startswith("/feedback "):
                parts = user_input[10:].strip().split(maxsplit=1)
                rating = parts[0]
                note = parts[1] if len(parts) > 1 else ""
                print("[Agent Memory]>", brain.feedback(rating, note))
                continue
            
            # --- Brain Processing Phase ---
            print("[Agent is thinking...]")
            
            # 1. Ask memory for relevant context
            rules = brain.recall(user_input, limit=2, kind="rule")
            goals = brain.recall("активная цель", limit=1, kind="goal")
            
            context_str = ""
            if goals:
                context_str += f"# Goal: {goals[0].text}\n"
            if rules:
                for r in rules:
                    context_str += f"# Rule: {r.text}\n"
            
            # Formulate the final prompt for the neural network
            # We add a newline to separate context from the actual user request
            final_prompt = context_str + "\n" + user_input if context_str else user_input
            
            if context_str:
                print(f"[Memory Injected]:\n{context_str.strip()}")
            
            # --- Generation Phase ---
            if model is None:
                print("[Agent]> Cannot generate code, neural network not loaded.")
                continue
                
            input_ids = enc.encode_ordinary(final_prompt)
            idx = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                # We stop generating when we see <|endoftext|> (token 50256 in gpt2)
                out_idx = model.generate(idx, max_new_tokens=150, temperature=0.7, top_k=40)
            
            # Decode the generated tokens only (skip the prompt)
            generated_tokens = out_idx[0][len(input_ids):].tolist()
            
            # Cut off the text if EOT token is found so it doesn't hallucinate next documents
            if enc.eot_token in generated_tokens:
                eot_index = generated_tokens.index(enc.eot_token)
                generated_tokens = generated_tokens[:eot_index]
                
            generated_text = enc.decode(generated_tokens)
            
            print("\n[Agent Code Output]⬇️")
            print(generated_text.strip())
            
            # Save the interaction to memory
            brain.remember(f"User asked: {user_input}. Agent generated code.", kind="dialogue")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
