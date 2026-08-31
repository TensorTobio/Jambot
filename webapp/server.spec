# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the local demo server.

Build from the repo root:  pyinstaller webapp/server.spec

The generated default missed every project package. ``evaluator`` is a
top-level import that only resolves once the repo root is on sys.path, and
``starter``/``api_call_agent`` are imported lazily inside ``AppState.agent()``,
so PyInstaller cannot see them by static analysis at all.
"""

from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # SPECPATH is webapp/

# catalog.jsonl is ~58 MB. Set to False to leave it out and ship a data/
# directory beside server.exe instead - server.py prefers that copy when present.
INCLUDE_CATALOG = True

datas = [
    (str(ROOT / "webapp" / "index.html"), "webapp"),
    (str(ROOT / "data" / "public_set.jsonl"), "data"),
]
if INCLUDE_CATALOG:
    datas.append((str(ROOT / "data" / "catalog.jsonl"), "data"))

a = Analysis(
    [str(ROOT / "webapp" / "server.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'evaluator',
        'evaluator.local_evaluator',
        'starter',
        'starter.agent',
        'starter.agent_keyword',
        'starter.retrieval',
        'starter.dialog',
        'api_call_agent',
        'api_call_agent.agent',
        'api_call_agent.llm_client',
        'api_call_agent.rephrase',
        'api_call_agent.rerank',
        'api_call_agent.reply',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server',
)
