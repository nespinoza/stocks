"""News pipeline: timestamped sources → structured LLM extraction → point-in-time signals."""
from stock_api.news.aggregate import (assign_sessions, cross_sectional_z, daily_panel, decay, dedupe, etf_signal,
                                      item_scores, normalize_title, variance_bump)
from stock_api.news.extraction import (PROMPT_VERSION, SCHEMA, AnthropicBackend, ClaudeCodeBackend, ExtractionCache,
                                       MockBackend, UsageLimitError, extract, item_key)
from stock_api.news.lexicon import load_lm, tone
