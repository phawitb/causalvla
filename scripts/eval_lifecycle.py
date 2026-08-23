"""Episode lifecycle bridge for LeRobot observation processors."""


def initialize_episode_processors(pipeline, task_id: int, episode_indices: list[int]) -> None:
    for step in pipeline.steps:
        begin_rollout = getattr(step, "begin_rollout", None)
        if begin_rollout is not None:
            begin_rollout(task_id=task_id, episode_indices=episode_indices)
