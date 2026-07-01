"""
Development server with hot-reload support.

Replicates lightrag_server.main() exactly but uses uvicorn's string-import
+ factory pattern so that ``--reload`` actually watches lightrag/*.py and
auto-restarts the server on every code change.

Usage:
    python debug_server.py                # manual run
    Or use VS Code: "LightRAG Server (dev reload)"
"""
import sys
import os

# ── Windows: force UTF-8 I/O so emoji / splash screen don't crash ──
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Windows event-loop fix (must run before any asyncio usage) ──────
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure project root is importable
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def create_app():
    """No-arg factory that uvicorn calls (once per reload child).

    Mirrors the app-construction portion of lightrag_server.main() so the
    child always picks up the latest code after a reload.
    """
    from lightrag.api.config import (
        initialize_config,
        get_config,
        update_uvicorn_mode_config,
    )
    from lightrag.api.lightrag_server import (
        create_app as _create_app,
        configure_logging,
    )

    initialize_config()
    configure_logging()
    update_uvicorn_mode_config()

    return _create_app(get_config())


def main():
    """One-time setup (parent process only), then hand off to uvicorn reloader."""

    from lightrag.api.config import initialize_config, global_args
    from lightrag.api.lightrag_server import check_and_install_dependencies
    from lightrag.api.utils_api import check_env_file, display_splash_screen

    # 1. Parse env + CLI args
    initialize_config()

    # 2. Guard: .env must exist
    if not check_env_file():
        sys.exit(1)

    # 3. Auto-install missing dependencies (pipmaster) — parent only
    check_and_install_dependencies()

    # 4. Required on Windows when multiprocessing is used (uvicorn reload)
    from multiprocessing import freeze_support

    freeze_support()

    # 5. Splash screen — parent only (child re-runs create_app which logs silently)
    display_splash_screen(global_args)

    # 6. Launch uvicorn reloader — pass the app as an import string so that
    #    uvicorn can re-import ``create_app`` in each reload child.
    import uvicorn

    print(
        f"Starting Uvicorn server (dev reload) on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(
        "debug_server:create_app",
        factory=True,
        host=global_args.host,
        port=global_args.port,
        log_config=None,
        reload=True,
        reload_dirs=[os.path.join(_project_root, "lightrag")],
    )


if __name__ == "__main__":
    main()
