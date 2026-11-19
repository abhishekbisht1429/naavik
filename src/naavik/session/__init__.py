"""Session lifecycle, configuration, state, and concrete workflow APIs."""

from naavik.session.base import Session
from naavik.session.registry import (
    register_session_type,
    session_class_for_type,
)
from naavik.session.analysis import AnalysisSession
from naavik.session.check import ConfigCheck, check_session_config
from naavik.session.components import (
    ActivationPlan,
    PlannedComponent,
    SessionComponents,
)
from naavik.session.config import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    SessionConfig,
    SessionPhase,
    normalize_session_config,
    normalize_session_type,
)
from naavik.session.training import TrainingSession

__all__ = [
    "ANALYSIS_SESSION_TYPE",
    "TRAINING_SESSION_TYPE",
    "ActivationPlan",
    "AnalysisSession",
    "ConfigCheck",
    "PlannedComponent",
    "Session",
    "SessionComponents",
    "SessionConfig",
    "SessionPhase",
    "TrainingSession",
    "check_session_config",
    "normalize_session_config",
    "normalize_session_type",
    "register_session_type",
    "session_class_for_type",
]
