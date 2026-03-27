from __future__ import annotations

try:
    from .app_ardu import main
except ImportError:
    from app_ardu import main


if __name__ == "__main__":
    main()
