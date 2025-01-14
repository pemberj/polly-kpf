"""
polly

kpf

Parameters specific to the KPF spectrograph and data filesystem
"""

from pathlib import Path


THORIUM_ORDER_INDICES = [*list(range(12)), 35]
LFC_ORDER_INDICES = [i for i in range(67) if i not in THORIUM_ORDER_INDICES]

ORDERLETS = ["SCI1", "SCI2", "SCI3", "CAL", "SKY"]
TIMESOFDAY = ["morn", "eve", "night", "midnight"]

MASTERS_DIR = Path("/data/kpf/masters")
L1_DIR = Path("/data/kpf/L1")
