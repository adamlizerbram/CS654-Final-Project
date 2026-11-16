# Track 5 safety shield

`agents.laya_shield:LayaShieldAgent` implements `act(env, obs) -> int` and
the evaluator's `threshold`, `overrides`, `last_override`, and `reset()` hooks.
The benchmark constructor accepts `dqn_path`, `laya_model`, and `threshold`.
The default adapters load an SB3 DQN and a Laya model; they have not yet been
validated with the team's trained models.

For tests or integration, inject `q_values(env, obs)`, returning one finite
Q-value per discrete action ID, and `unsafe_score(env, obs, action)`, returning
a finite score in [0, 1]. The default Laya adapter uses the shared
`describe_with_action` and `safety_question` helpers and reads the `noul`
unsafe score. The team's DQN checkpoint and observation compatibility must be
confirmed before relying on the SB3 adapter. The merged Laya driving agent
provides action choices, not a shield score API.

The shield orders legal actions by descending Q-value, breaking ties by action
ID, then selects the first with score <= threshold. If none pass, it selects
the lowest estimated risk, with Q rank breaking ties. It counts a change from
the DQN's unrestricted preferred action as an override, including when that
action is unavailable. A fallback or override does not prove safety or a
prevented collision. Scores may be uncalibrated; thresholds near 0 and 1 are
useful candidates for later experiments, not established operating points.

Run focused tests from the project root:

```sh
python -m unittest discover -s tests -p 'test_laya_shield.py' -v
```
