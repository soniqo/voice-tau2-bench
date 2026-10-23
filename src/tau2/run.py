"""Task-loading helpers; voice execution is exposed by tau-voice."""

from tau2.registry import registry


def get_tasks(
    task_set_name, task_split_name=None, task_ids=None, num_tasks=None, task_kwargs=None
):
    tasks = registry.get_tasks_loader(task_set_name)(
        task_split_name=task_split_name, **(task_kwargs or {})
    )
    if task_ids is not None:
        selected = [task for task in tasks if task.id in task_ids]
        missing = set(task_ids) - {task.id for task in selected}
        if missing:
            raise ValueError(f"Not all tasks were found: {sorted(missing)}")
        tasks = selected
    return tasks if num_tasks is None else tasks[:num_tasks]
