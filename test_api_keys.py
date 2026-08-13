"""
test_api_keys.py
=================
Quick smoke test: verify that both With-Sentiment and Without-Sentiment
strategies correctly load their dedicated Gemini API keys from .env files.

Run from repo root:
    python test_api_keys.py
"""
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
WITH_DIR    = os.path.join(REPO_ROOT, "With LLM", "Intraday_Cross_Sectional_Mean_Reversion_With_Sentiment")
WITHOUT_DIR = os.path.join(REPO_ROOT, "With LLM", "Intraday_Cross_Sectional_Mean_Reversion_Without_Sentiment")

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(condition)


def load_env(env_path: str):
    """Parse a .env file and return a dict of key->value."""
    env = {}
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


# ──────────────────────────────────────────────────────────────────────────────
# 1. Check .env files contain the correct variable names
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- 1. .env file variable names ---")

with_env_path    = os.path.join(WITH_DIR,    ".env")
without_env_path = os.path.join(WITHOUT_DIR, ".env")

check(".env exists (With Sentiment)",    os.path.isfile(with_env_path),    with_env_path)
check(".env exists (Without Sentiment)", os.path.isfile(without_env_path), without_env_path)

with_env    = load_env(with_env_path)    if os.path.isfile(with_env_path)    else {}
without_env = load_env(without_env_path) if os.path.isfile(without_env_path) else {}

check("GEMINI_API_KEY_WITH_SENTIMENT present",
      "GEMINI_API_KEY_WITH_SENTIMENT" in with_env)
check("GEMINI_API_KEY_WITH_SENTIMENT not empty",
      bool(with_env.get("GEMINI_API_KEY_WITH_SENTIMENT")))
check("Old GEMINI_API_KEY NOT present in With-Sentiment .env",
      "GEMINI_API_KEY" not in with_env)

check("GEMINI_API_KEY_WITHOUT_SENTIMENT present",
      "GEMINI_API_KEY_WITHOUT_SENTIMENT" in without_env)
check("GEMINI_API_KEY_WITHOUT_SENTIMENT not empty",
      bool(without_env.get("GEMINI_API_KEY_WITHOUT_SENTIMENT")))
check("Old GEMINI_API_KEY NOT present in Without-Sentiment .env",
      "GEMINI_API_KEY" not in without_env)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Simulate config.py env-var lookup (fallback chain)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- 2. config.py env-var lookup simulation ---")

os.environ["GEMINI_API_KEY_WITH_SENTIMENT"] = with_env.get("GEMINI_API_KEY_WITH_SENTIMENT", "")

resolved_with = (
    os.getenv("GEMINI_API_KEY_WITH_SENTIMENT")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
)
check("With-Sentiment: api_key resolved", bool(resolved_with),
      f"key starts with {resolved_with[:10]}..." if resolved_with else "EMPTY")

del os.environ["GEMINI_API_KEY_WITH_SENTIMENT"]
os.environ["GEMINI_API_KEY_WITHOUT_SENTIMENT"] = without_env.get("GEMINI_API_KEY_WITHOUT_SENTIMENT", "")

resolved_without = (
    os.getenv("GEMINI_API_KEY_WITHOUT_SENTIMENT")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
)
check("Without-Sentiment: api_key resolved", bool(resolved_without),
      f"key starts with {resolved_without[:10]}..." if resolved_without else "EMPTY")

check("Two keys are DIFFERENT (expected)", resolved_with != resolved_without,
      "SAME key – defeats the purpose!" if resolved_with == resolved_without else "distinct keys confirmed")

del os.environ["GEMINI_API_KEY_WITHOUT_SENTIMENT"]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Import ai_decision.config and run LLMConfig.load() from each folder
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- 3. Import & load LLMConfig from each folder ---")

import importlib

# ── With Sentiment ──
os.environ["GEMINI_API_KEY_WITH_SENTIMENT"] = with_env.get("GEMINI_API_KEY_WITH_SENTIMENT", "")
sys.path.insert(0, WITH_DIR)
try:
    import ai_decision.config as wsc_module
    importlib.reload(wsc_module)
    cfg_with = wsc_module.LLMConfig.load(os.path.join(WITH_DIR, "config", "config.yaml"))
    check("LLMConfig.load() succeeded (With Sentiment)", True)
    check("api_key set on LLMConfig (With Sentiment)", bool(cfg_with.api_key),
          f"starts with {cfg_with.api_key[:10]}..." if cfg_with.api_key else "EMPTY")
except Exception as exc:
    check("LLMConfig.load() succeeded (With Sentiment)", False, str(exc))
    check("api_key set on LLMConfig (With Sentiment)", False)
finally:
    sys.path.pop(0)
    del os.environ["GEMINI_API_KEY_WITH_SENTIMENT"]

for mod in list(sys.modules.keys()):
    if "ai_decision" in mod:
        del sys.modules[mod]

# ── Without Sentiment ──
os.environ["GEMINI_API_KEY_WITHOUT_SENTIMENT"] = without_env.get("GEMINI_API_KEY_WITHOUT_SENTIMENT", "")
sys.path.insert(0, WITHOUT_DIR)
try:
    import ai_decision.config as woc_module
    cfg_without = woc_module.LLMConfig.load(os.path.join(WITHOUT_DIR, "config", "config.yaml"))
    check("LLMConfig.load() succeeded (Without Sentiment)", True)
    check("api_key set on LLMConfig (Without Sentiment)", bool(cfg_without.api_key),
          f"starts with {cfg_without.api_key[:10]}..." if cfg_without.api_key else "EMPTY")
except Exception as exc:
    check("LLMConfig.load() succeeded (Without Sentiment)", False, str(exc))
    check("api_key set on LLMConfig (Without Sentiment)", False)
finally:
    sys.path.pop(0)
    del os.environ["GEMINI_API_KEY_WITHOUT_SENTIMENT"]


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print()
total  = len(results)
passed = sum(results)
failed = total - passed
print(f"=== Results: {passed}/{total} passed", "- All good!" if failed == 0 else f"- {failed} FAILED — check above")
sys.exit(0 if failed == 0 else 1)
