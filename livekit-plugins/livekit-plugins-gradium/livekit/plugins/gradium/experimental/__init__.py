"""Experimental Gradium STT with client-side end-of-turn detection.

`signals`, `scorer`, `machine` and `profiles` have no network I/O, so the offline
simulator and the eot-bench adapter import them directly.
"""

from .machine import TurnMachine, TurnState
from .profiles import Eagerness, TurnOverrides, TurnProfile, preset
from .stt import SpeechStream, STTExperimental

__all__ = [
    "Eagerness",
    "STTExperimental",
    "SpeechStream",
    "TurnMachine",
    "TurnOverrides",
    "TurnProfile",
    "TurnState",
    "preset",
]
