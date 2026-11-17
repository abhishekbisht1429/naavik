"""Session lifecycle, configuration, state, and concrete workflow APIs."""

from training_framework.session.base import Session
from training_framework.session.registry import (
    register_session_type,
    session_class_for_type,
)
from training_framework.session.analysis import AnalysisSession
from training_framework.session.check import ConfigCheck, check_session_config
from training_framework.session.components import (
    ActivationPlan,
    PlannedComponent,
    SessionComponents,
)
from training_framework.session.config import (
    ANALYSIS_SESSION_TYPE,
    TRAINING_SESSION_TYPE,
    SessionConfig,
    SessionPhase,
    normalize_session_config,
    normalize_session_type,
)
from training_framework.session.training import TrainingSession

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
