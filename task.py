import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

Task = Dict[str, Tuple[int, datetime, str]]


def read_tasks_csv(path_tasks: Path) -> list[Task]:
    """
    reads a csv file to a list of tasks

    task example:
    id, name, start, end,  prio, duration
    0,do something,2024-04-15 08:00:00,2024-04-15 14:00:00,1,2
    """
    with open(path_tasks.as_posix(), 'r') as f:
        reader = csv.DictReader(
            f, fieldnames=["id", "name", "start", "end",  "prio", "duration"]
        )
        tasks = []
        format: str = "%Y-%m-%d %H:%M:%S"
        i: int = 0
        for row in reader:
            start = datetime.strptime(row["start"], format)
            end = datetime.strptime(row["end"], format)
            duration = int(row['duration'])

            row['id'] = int(row['id'])
            row['prio'] = int(row['prio'])
            row['start'] = start
            row['end'] = end
            row['duration'] = duration

            # basic checks for valid tasks
            assert start <= end
            assert (end - start + timedelta(hours=1)).seconds // 3600 >= duration
            assert 8 <= start.hour <= 18 and 8 <= end.hour <= 18
            i += 1

            tasks.append(row)

        assert len(set(map(lambda x: x['id'], tasks))) == i
    return tasks
