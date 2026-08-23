from scripts.eval_lifecycle import initialize_episode_processors


class _Step:
    def __init__(self):
        self.calls = []

    def begin_rollout(self, task_id, episode_indices):
        self.calls.append((task_id, episode_indices))


class _Pipeline:
    def __init__(self, steps):
        self.steps = steps


def test_initializes_episode_aware_steps_and_ignores_plain_steps():
    aware = _Step()
    initialize_episode_processors(_Pipeline([object(), aware]), task_id=4, episode_indices=[10, 11])
    assert aware.calls == [(4, [10, 11])]
