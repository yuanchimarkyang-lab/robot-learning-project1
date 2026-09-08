from lerobot.datasets import LeRobotDatasetMetadata

repo_id = "lerobot/libero"

meta = LeRobotDatasetMetadata(repo_id)

print(meta)
print("\nFPS:")
print(meta.fps)

print("\nFeatures:")
for key, value in meta.features.items():
    print(key, value)

print("\nTasks:")
for task in meta.tasks:
    print(task)

tasks = meta.tasks.reset_index().sort_values("task_index")
print(tasks.to_string(index=False))


print(type(meta.tasks))
print(meta.tasks.index)
print(meta.tasks.head())
