from fastapi import APIRouter, Request

from memory_service import get_memory_usage, get_gpu_memory_usage
from pyutils.logs import llog
from tts_service import change_sovits_weights, change_gpt_weights, handle_control, handle_change, handle

index_router = APIRouter()

@index_router.post("/set_model")
async def set_model(request: Request):
  json_post_raw = await request.json()
  global gpt_path
  gpt_path = json_post_raw.get("gpt_model_path")
  global sovits_path
  sovits_path = json_post_raw.get("sovits_model_path")
  llog.info("gptpath" + gpt_path + ";vitspath" + sovits_path)
  change_sovits_weights(sovits_path)
  change_gpt_weights(gpt_path)
  return "ok"


@index_router.post("/control")
async def control(request: Request):
  json_post_raw = await request.json()
  return handle_control(json_post_raw.get("command"))


@index_router.get("/control")
async def control(command: str = None):
  return handle_control(command)


@index_router.post("/change_refer")
async def change_refer(request: Request):
  json_post_raw = await request.json()
  return handle_change(
    json_post_raw.get("refer_wav_path"),
    json_post_raw.get("prompt_text"),
    json_post_raw.get("prompt_language")
  )


@index_router.get("/change_refer")
async def change_refer(
  refer_wav_path: str = None,
  prompt_text: str = None,
  prompt_language: str = None
):
  return handle_change(refer_wav_path, prompt_text, prompt_language)


@index_router.post("/")
async def tts_endpoint(request: Request):
  json_post_raw = await request.json()
  return handle(
    json_post_raw.get("refer_wav_path"),
    json_post_raw.get("prompt_text"),
    json_post_raw.get("prompt_language"),
    json_post_raw.get("text"),
    json_post_raw.get("text_language"),
    json_post_raw.get("cut_punc"),
  )


@index_router.get("/")
async def tts_endpoint(
  refer_wav_path: str = None,
  prompt_text: str = None,
  prompt_language: str = None,
  text: str = None,
  text_language: str = None,
  cut_punc: str = None,
):
  return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, cut_punc)

@index_router.get("/memory-usage")
def read_memory_usage():
    memory_usage = get_memory_usage()
    return {"memory_usage": memory_usage}

@index_router.get("/gpu-memory-usage")
def read_gpu_memory_usage():
    gpu_memory_usage = get_gpu_memory_usage()
    return {"gpu_memory_usage": gpu_memory_usage}