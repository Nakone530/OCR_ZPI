import logging

from logging import info  # noqa: F401  – info dostępne dla całego pakietu

logging.getLogger(__name__).addHandler(logging.NullHandler())
