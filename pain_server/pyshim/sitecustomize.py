# Loaded at interpreter start when this folder is on PYTHONPATH (run.bat and install.bat put it there).
#
# On some Windows machines the WMI service hangs. Python 3.12's platform module asks WMI for the OS version and
# CPU name (platform.uname(), platform.machine(), win32_ver()), and numpy, pandas, aiohttp, torch and pip all call
# it during import, so a hung WMI freezes them with no output. Disabling the WMI query makes platform fall back to
# sys.getwindowsversion() and the PROCESSOR_* environment variables, which is what Python 3.11 and earlier did.
# Where WMI works this changes nothing visible.
import sys

if sys.platform == "win32":
    import platform

    def _no_wmi(*a, **k):
        raise OSError("WMI disabled by pyshim")

    if hasattr(platform, "_wmi_query"):
        platform._wmi_query = _no_wmi
