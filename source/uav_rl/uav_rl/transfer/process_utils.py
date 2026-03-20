from __future__ import annotations

import os
import signal
import subprocess
import time


def matching_ardupilot_process(cmdline: str) -> bool:
    cmdline_lower = cmdline.lower()
    return "sim_vehicle.py" in cmdline_lower or "mavproxy.py" in cmdline_lower or "arducopter" in cmdline_lower


def get_process_table() -> dict[int, dict[str, int | str]]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.SubprocessError:
        return {}

    table: dict[int, dict[str, int | str]] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = {"ppid": ppid, "cmdline": parts[2]}
    return table


def descendants(process_table: dict[int, dict[str, int | str]], root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, info in process_table.items():
        children.setdefault(int(info["ppid"]), []).append(pid)

    ordered: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        stack.extend(children.get(pid, []))
    return list(reversed(ordered))


def terminate_process_tree(root_pid: int, timeout: float = 5.0, process_table=None) -> int:
    if process_table is None:
        process_table = get_process_table()

    if root_pid not in process_table:
        return 0

    targets = descendants(process_table, root_pid)
    if not targets:
        return 0

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGINT)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [pid for pid in targets if os.path.exists(f"/proc/{pid}")]
        if not alive:
            return len(targets)
        time.sleep(0.1)

    try:
        os.killpg(os.getpgid(root_pid), signal.SIGKILL)
    except OSError:
        for pid in targets:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    time.sleep(0.2)
    return len(targets)


def cleanup_ardupilot_processes() -> int:
    process_table = get_process_table()
    root_pids = [pid for pid, info in process_table.items() if matching_ardupilot_process(str(info["cmdline"]))]

    cleaned = 0
    for pid in sorted(set(root_pids)):
        cleaned += terminate_process_tree(pid, timeout=2.0, process_table=process_table)
    return cleaned
