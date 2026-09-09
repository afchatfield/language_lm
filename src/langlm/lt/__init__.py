"""LanguageTool integration: the source of rule ids and explanation text."""

from langlm.lt.client import LanguageToolClient, LanguageToolError, Match

__all__ = ["LanguageToolClient", "LanguageToolError", "Match"]
