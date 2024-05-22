from fastapi import FastAPI

from pyutils.logs import llog

llog.info("start server")
app = FastAPI()
