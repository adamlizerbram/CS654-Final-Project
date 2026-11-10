"""Convert highway-env state into text and questions for Laya.

Reads the simulator's internal state (env.unwrapped) rather than the
normalized observation array, so the text has real units (meters, m/s).
Tested with highway-env 1.12.1.
"""

ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

ACTION_TEXT = {
    "LANE_LEFT": "change one lane to the left",
    "IDLE": "keep the current lane and speed",
    "LANE_RIGHT": "change one lane to the right",
    "FASTER": "speed up in the current lane",
    "SLOWER": "slow down in the current lane",
}

SENSING_RANGE = 80.0  # meters; ignore vehicles farther away than this


def _gap_and_closing(ego, other):
    """Bumper-to-bumper gap (m) and closing speed (m/s, positive = getting closer)."""
    dx = other.position[0] - ego.position[0]
    gap = abs(dx) - ego.LENGTH
    closing = (ego.speed - other.speed) if dx > 0 else (other.speed - ego.speed)
    return max(gap, 0.0), closing


def _describe_neighbor(ego, other, where):
    if other is None:
        return f"{where}: clear"
    gap, closing = _gap_and_closing(ego, other)
    if gap > SENSING_RANGE:
        return f"{where}: clear"
    text = f"{where}: car {gap:.0f} m away, {other.speed:.0f} m/s"
    if closing > 0.5:
        ttc = gap / closing
        text += f", closing at {closing:.0f} m/s (collision in {ttc:.1f} s if nothing changes)"
    elif closing < -0.5:
        text += ", pulling away"
    else:
        text += ", same speed"
    return text


def describe_state(env):
    """Short, fixed-format description of the situation around the ego car."""
    u = env.unwrapped
    ego, road = u.vehicle, u.road
    lanes = u.config["lanes_count"]
    _, _, lane_id = ego.lane_index

    lines = [
        f"Highway with {lanes} lanes, numbered 1 (leftmost) to {lanes} (rightmost).",
        f"Ego car: lane {lane_id + 1}, speed {ego.speed:.0f} m/s.",
    ]

    # Current lane, then the adjacent lanes if they exist.
    for side, offset in (("Current lane", 0), ("Left lane", -1), ("Right lane", 1)):
        target = lane_id + offset
        if not 0 <= target < lanes:
            lines.append(f"{side}: does not exist.")
            continue
        lane_index = (ego.lane_index[0], ego.lane_index[1], target)
        front, rear = road.neighbour_vehicles(ego, lane_index)
        lines.append(f"{side}: " + "; ".join([
            _describe_neighbor(ego, front, "ahead"),
            _describe_neighbor(ego, rear, "behind"),
        ]) + ".")

    return " ".join(lines)


def available_actions(env):
    """Action names that are legal right now (e.g. no LANE_LEFT in the leftmost lane)."""
    return [ACTIONS[i] for i in env.unwrapped.action_type.get_available_actions()]


def action_choice_question(env):
    """Laya-alone: one choice question over the currently legal actions."""
    return {
        "action": {
            "type": "choice",
            "instructions": "Which action is safest while still making good progress?",
            "criteria": {a: ACTION_TEXT[a] for a in available_actions(env)},
        }
    }


def safety_question():
    """Shield: yes/no probability that a proposed action is unsafe.

    Ask it with describe_with_action(), once per candidate action.
    """
    return {
        "unsafe": {
            "type": "noul",
            "instructions": "Will the proposed action lead to a collision "
                            "or a dangerously small gap within the next 3 seconds?",
        }
    }


def describe_with_action(env, action_name):
    return describe_state(env) + f" Proposed action: {ACTION_TEXT[action_name]}."


if __name__ == "__main__":
    import gymnasium as gym
    import highway_env  # noqa: F401  (registers the environments)

    env = gym.make("highway-v0")
    env.reset(seed=0)
    for _ in range(5):
        env.step(1)  # IDLE a few steps so traffic spreads out
    print(describe_state(env))
    print("Legal actions:", available_actions(env))
    print(describe_with_action(env, "LANE_LEFT"))
