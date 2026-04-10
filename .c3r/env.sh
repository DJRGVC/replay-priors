# Sourced by agent_loop.sh at the start of each iteration.
# Add any env vars, venv activations, or CUDA settings this project needs.
export WANDB_API_KEY=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
# DISABLED: causes claude -p to route through API (10k tok/min limit) instead of Claude.ai
# [ -f "$HOME/.anthropic_key" ] && export ANTHROPIC_API_KEY="$(cat "$HOME/.anthropic_key")"
unset ANTHROPIC_API_KEY 2>/dev/null
