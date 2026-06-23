__version__ = "1.0.0"
from .vbot_robot_data import PinVBotModel

__all__ = ["PinVBotModel"]

try:
    from .mujoco_vbot_model import MuJoCo_VBot_Model
except ModuleNotFoundError as exc:
    if exc.name != "mujoco":
        raise
else:
    __all__.append("MuJoCo_VBot_Model")
