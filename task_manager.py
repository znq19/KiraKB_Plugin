import asyncio
import uuid
from enum import Enum
from typing import Dict, Callable, Optional

from core.logging_manager import get_logger

logger = get_logger("kb_task", "cyan")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task:
    def __init__(self, task_id: str, kb_id: str, description: str, total_steps: int = 0):
        self.task_id = task_id
        self.kb_id = kb_id
        self.description = description
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.current_step = 0
        self.total_steps = total_steps
        self.message = ""
        self.result = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "kb_id": self.kb_id,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result
        }


class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def create_task(self, kb_id: str, description: str, total_steps: int = 0) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = Task(task_id, kb_id, description, total_steps)
        self.tasks[task_id] = task
        return task_id

    async def run_task(self, task_id: str, coro: Callable, on_progress: Optional[Callable] = None):
        task = self.tasks.get(task_id)
        if not task:
            return
        task.status = TaskStatus.RUNNING
        try:
            async def progress_callback(current, total, step_desc=""):
                task.current_step = current
                task.total_steps = total
                task.progress = int(current / total * 100) if total > 0 else 0
                task.message = step_desc
                if on_progress:
                    await on_progress(task_id, task.progress, step_desc)
            result = await coro(progress_callback)
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.result = result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.message = str(e)
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_tasks_for_kb(self, kb_id: str) -> list:
        return [t.to_dict() for t in self.tasks.values() if t.kb_id == kb_id]


_task_manager = None

def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager