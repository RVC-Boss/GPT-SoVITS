import sys
from typing import Optional

from loguru import logger
from rich.console import Console, JustifyMethod
from rich.highlighter import Highlighter
from rich.logging import RichHandler
from rich.progress import Task, TextColumn
from rich.style import StyleType
from rich.table import Column
from rich.text import Text
from rich.traceback import Traceback, install

console = Console(stderr=False)
install(console=console)


def loguru_format(record):
    level = record["level"].name
    color = {
        "DEBUG": "green",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bright_red",
    }.get(level, "white")

    return f"[bold {color}][{level}][/bold {color}] " + "{message}"


handler_with_locals = RichHandler(
    console=console,
    show_time=False,
    show_path=False,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    show_level=False,
    markup=True,
)
handler_without_locals = RichHandler(
    console=console,
    show_time=False,
    show_path=False,
    rich_tracebacks=True,
    tracebacks_show_locals=False,
    show_level=False,
    markup=True,
)


def local_filter(r):
    return r["extra"].get("show_locals", True)


logger.remove()
logger.add(handler_with_locals, format=loguru_format, filter=local_filter)
logger.add(handler_without_locals, format=loguru_format, filter=lambda x: not local_filter(x))


class SpeedColumnToken(TextColumn):
    """Show task progress as a percentage.

    Args:
        text_format (str, optional): Format for percentage display. Defaults to "[progress.percentage]{task.percentage:>3.0f}%".
        text_format_no_percentage (str, optional): Format if percentage is unknown. Defaults to "".
        style (StyleType, optional): Style of output. Defaults to "none".
        justify (JustifyMethod, optional): Text justification. Defaults to "left".
        markup (bool, optional): Enable markup. Defaults to True.
        highlighter (Optional[Highlighter], optional): Highlighter to apply to output. Defaults to None.
        table_column (Optional[Column], optional): Table Column to use. Defaults to None.
        show_speed (bool, optional): Show speed if total is unknown. Defaults to False.
    """

    def __init__(
        self,
        text_format: str = "[progress.percentage]{task.percentage:>3.0f}%",
        text_format_no_percentage: str = "",
        style: StyleType = "none",
        justify: JustifyMethod = "left",
        markup: bool = True,
        highlighter: Optional[Highlighter] = None,
        table_column: Optional[Column] = None,
        show_speed: bool = True,
    ) -> None:
        self.text_format_no_percentage = text_format_no_percentage
        self.show_speed = show_speed
        super().__init__(
            text_format=text_format,
            style=style,
            justify=justify,
            markup=markup,
            highlighter=highlighter,
            table_column=table_column,
        )

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        """Render the speed in iterations per second.

        Args:
            task (Task): A Task object.

        Returns:
            Text: Text object containing the task speed.
        """
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.1f} token/s", style="progress.percentage")

    def render(self, task: Task) -> Text:
        if self.show_speed:
            return self.render_speed(task.finished_speed or task.speed)
        text_format = self.text_format_no_percentage if task.total is None else self.text_format
        _text = text_format.format(task=task)
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


class SpeedColumnIteration(TextColumn):
    """Show task progress as a percentage.

    Args:
        text_format (str, optional): Format for percentage display. Defaults to "[progress.percentage]{task.percentage:>3.0f}%".
        text_format_no_percentage (str, optional): Format if percentage is unknown. Defaults to "".
        style (StyleType, optional): Style of output. Defaults to "none".
        justify (JustifyMethod, optional): Text justification. Defaults to "left".
        markup (bool, optional): Enable markup. Defaults to True.
        highlighter (Optional[Highlighter], optional): Highlighter to apply to output. Defaults to None.
        table_column (Optional[Column], optional): Table Column to use. Defaults to None.
        show_speed (bool, optional): Show speed if total is unknown. Defaults to False.
    """

    def __init__(
        self,
        text_format: str = "[progress.percentage]{task.percentage:>3.0f}%",
        text_format_no_percentage: str = "",
        style: StyleType = "none",
        justify: JustifyMethod = "left",
        markup: bool = True,
        highlighter: Optional[Highlighter] = None,
        table_column: Optional[Column] = None,
        show_speed: bool = True,
    ) -> None:
        self.text_format_no_percentage = text_format_no_percentage
        self.show_speed = show_speed
        super().__init__(
            text_format=text_format,
            style=style,
            justify=justify,
            markup=markup,
            highlighter=highlighter,
            table_column=table_column,
        )

    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        """Render the speed in iterations per second.

        Args:
            task (Task): A Task object.

        Returns:
            Text: Text object containing the task speed.
        """
        if speed is None:
            return Text("", style="progress.percentage")
        return Text(f"{speed:.1f} it/s", style="progress.percentage")

    def render(self, task: Task) -> Text:
        if self.show_speed:
            return self.render_speed(task.finished_speed or task.speed)
        text_format = self.text_format_no_percentage if task.total is None else self.text_format
        _text = text_format.format(task=task)
        if self.markup:
            text = Text.from_markup(_text, style=self.style, justify=self.justify)
        else:
            text = Text(_text, style=self.style, justify=self.justify)
        if self.highlighter:
            self.highlighter.highlight(text)
        return text


def tb(show_locals: bool = True):
    exc_type, exc_value, exc_tb = sys.exc_info()
    assert exc_type
    assert exc_value
    tb = Traceback.from_exception(exc_type, exc_value, exc_tb, show_locals=show_locals)

    return tb


__all__ = ["logger", "console", "tb", "SpeedColumnToken", "SpeedColumnIteration"]

if __name__ == "__main__":
    try:
        raise RuntimeError()
    except Exception:
        logger.bind(show_locals=False).exception("TEST")
