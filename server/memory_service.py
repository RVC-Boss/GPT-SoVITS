import psutil
import subprocess


def get_memory_usage():
  process = psutil.Process()
  mem_info = process.memory_info()
  memory_usage = {
    "rss": mem_info.rss / (1024 ** 2),  # Resident Set Size
    "vms": mem_info.vms / (1024 ** 2),  # Virtual Memory Size
    "percent": process.memory_percent()  # Percentage of memory usage
  }
  return memory_usage


def get_gpu_memory_usage():
  try:
    result = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True,
      text=True
    )
    output = result.stdout.strip()
    if output:
      used_memory, total_memory = map(int, output.split(', '))
      gpu_memory_usage = {
        "used_memory": used_memory,  # in MiB
        "total_memory": total_memory,  # in MiB
        "percent": (used_memory / total_memory) * 100  # Percentage of GPU memory usage
      }
      return gpu_memory_usage
    else:
      return {"error": "No GPU found or unable to query GPU memory usage."}
  except Exception as e:
    return {"error": str(e)}
