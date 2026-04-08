# INBOX ARCHIVE

---
[2026-04-08 04:24 UTC] Daniel G → td_baseline
MSG: hey, can you give me an idea of where you are starting?
RESP: Starting iter_001 — code scaffolding (env wrapper, train.py, td_instrumenter) already exists from initial setup. This iter: smoke-test the env + SB3 stack locally, fix the dense-reward oracle gap in td_instrumenter (SB3 doesn't store infos in replay buffer), then do a short local run. Will report back.

---
[2026-04-08 04:30 UTC] vlm_probe → td_baseline
MSG: Daniel says to use Modal for most runs — he has lots of credits and it should already be logged in. Use it for GPU/training workloads.
RESP: Will do — will set up Modal for training runs after smoke-testing locally.
