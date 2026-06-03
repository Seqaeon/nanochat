import torch
from nanochat.gpt import GPTConfig
from nanochat.eet import EarlyExitGPT

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = GPTConfig(
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        vocab_size=1000,
        sequence_len=128,
        n_layer=4,
        use_eet=True,
        eet_min_exit_layer=1,
        eet_compute_skip=True,
        eet_target_active_frac=0.10,
        eet_global_router=True,
        eet_gumbel_temp_start=1.0,
        eet_gumbel_hard=True,
        eet_reenter_final=True,
        eet_loss_variant='ce_guided',
    )

    model = EarlyExitGPT(config).to(device)
    model.train()

    B, T = 2, 128
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Let's use explanation tool to compile and find graph breaks
    explanation = torch._dynamo.explain(
        model,
        x, y,
        eet_do_route=True,
        eet_phase=3,
        eet_lambda_r=torch.tensor(0.0, device=device),
        eet_lambda_e=torch.tensor(0.1, device=device),
        eet_gumbel_temp=torch.tensor(1.0, device=device),
        eet_step=torch.tensor(0.0, device=device),
        eet_total_steps=torch.tensor(1000.0, device=device),
    )

    print("\n--- GRAPH BREAKS ---")
    print(f"Number of graph breaks: {len(explanation.graph_breaks)}")
    for i, gb in enumerate(explanation.graph_breaks):
        print(f"Graph break {i}:")
        print(f"  Reason: {gb.reason}")
        print(f"  User stack: {gb.user_stack}")

if __name__ == "__main__":
    main()
