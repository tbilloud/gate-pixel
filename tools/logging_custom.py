import sys
import logging

try:
    # opengate >= 10.0.3 — loguru-based
    from opengate.logger import logger as global_log

    # loguru's logger.remove() is called by opengate, so there are no sinks.
    # Add one so that messages are actually printed.
    if not global_log._core.handlers:
        global_log.add(sys.stderr, level="INFO")

except ImportError:
    # opengate 10.0.1 — colorlog / stdlib logging-based
    from opengate.logger import global_log

    global_log.setLevel(logging.INFO)
