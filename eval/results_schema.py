# One row per evaluation episode; threshold and overrides are empty for unshielded methods.
COLUMNS = [
    "method",       # e.g. dqn, laya_zeroshot, laya_finetuned, shield_zeroshot, shield_finetuned
    "model_seed",   # DQN training seed; empty for Laya alone
    "env_seed",     # from configs/seeds.yaml eval range
    "crashed",      # 0/1
    "steps",        # decisions taken before the episode ended
    "avg_speed",    # m/s
    "distance",     # m
    "reward",       # episode return
    "threshold",    # shield P(unsafe) threshold
    "overrides",    # times the shield replaced DQN's action
]
