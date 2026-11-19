from __future__ import annotations

from typing import TYPE_CHECKING

from naavik.components import hook
from naavik.components.builtin.observability import Logger

if TYPE_CHECKING:
    from naavik.session import Session


@hook("logger", session_type="analysis")
class AnalysisLogger(Logger):
    """Default progress logger for analysis sessions."""

    def pre_iteration_callback(self, session: Session) -> None:
        self.print(
            f"Analysis iteration {session.iteration}/"
            f"{session.session_config.max_iterations}"
        )
