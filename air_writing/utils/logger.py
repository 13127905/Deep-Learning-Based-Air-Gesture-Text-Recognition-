"""logger.py — Colour-coded console + file logger."""
import logging, os, sys
from datetime import datetime

class _C:
    R="\033[0m"; RED="\033[91m"; GRN="\033[92m"
    YLW="\033[93m"; CYN="\033[96m"; MGT="\033[95m"; BLD="\033[1m"

class _Fmt(logging.Formatter):
    _MAP={logging.DEBUG:_C.CYN,logging.INFO:_C.GRN,
          logging.WARNING:_C.YLW,logging.ERROR:_C.RED,logging.CRITICAL:_C.MGT+_C.BLD}
    def format(self,r):
        r.levelname=f"{self._MAP.get(r.levelno,'')}{r.levelname}{_C.R}"
        return super().format(r)

def get_logger(name,log_dir="logs",debug=False):
    log=logging.getLogger(name)
    if log.handlers: return log
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    ch=logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(_Fmt("%(asctime)s | %(levelname)s | %(name)s | %(message)s","%H:%M:%S"))
    log.addHandler(ch)
    os.makedirs(log_dir,exist_ok=True)
    fh=logging.FileHandler(os.path.join(log_dir,f"aw_{datetime.now():%Y%m%d}.log"),encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    log.addHandler(fh)
    return log
