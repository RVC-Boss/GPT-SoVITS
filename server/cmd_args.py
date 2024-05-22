class CmdArgs:
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(CmdArgs, cls).__new__(cls)
    return cls._instance

  def set_args(self, args):
    self.args = args

  def get_args(self):
    return self.args
