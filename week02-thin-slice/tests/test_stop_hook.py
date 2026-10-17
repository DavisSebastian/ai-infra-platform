import threading

import pytest


def test_stop_hook_reports_stop_only_after_the_flag_is_set():
    # Needs torch, which CI does not install, so this runs on your machine and shows as "skipped" in CI.
    torch = pytest.importorskip("torch")
    from app.model import StopWhenSet

    flag = threading.Event()
    hook = StopWhenSet(flag)
    batch_of_two = torch.zeros((2, 5), dtype=torch.long)

    assert hook(batch_of_two, None).tolist() == [False, False]
    flag.set()
    assert hook(batch_of_two, None).tolist() == [True, True]
