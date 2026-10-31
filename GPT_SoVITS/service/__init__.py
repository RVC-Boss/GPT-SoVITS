"""HTTP service components for the GPT-SoVITS inference application.

The package keeps imports lazy so importing a focused helper, such as an audio
packer, does not construct the application or inference configuration.
"""

__all__ = ["APP", "TTS_Request", "create_app"]


def __getattr__(name):
    if name == "TTS_Request":
        from .schemas import TTS_Request

        return TTS_Request
    if name in {"APP", "create_app"}:
        from .app import APP, create_app

        return {"APP": APP, "create_app": create_app}[name]
    raise AttributeError(name)
