from  us_visa.logger import logging
from us_visa.exception import USvisaException
import sys

try:
    a = 1 / 0

except Exception as e:
    # also store in logger file
    logging.error("An error occurred: %s", str(e))

    raise USvisaException(e, sys)