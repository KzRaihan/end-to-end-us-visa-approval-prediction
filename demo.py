from us_visa.exception import USvisaException
import os, sys

try:
    a = 1/0
except Exception as e:
    raise USvisaException(e, sys) from e