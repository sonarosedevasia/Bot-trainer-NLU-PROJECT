from .users import router as users
from .workspace import router as workspace
from .train import router as train
from .evaluate import router as evaluate
from .model_compare import router as model_compare
from .active_learning import router as active_learning   # ✅ NEW

__all__ = [
    "users",
    "workspace",
    "train",
    "evaluate",
    "model_compare",
    "active_learning",
]
