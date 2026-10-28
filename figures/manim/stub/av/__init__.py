# PyAV's DLLs are blocked by Windows Application Control on this laptop; still images never touch video, so a stub.
class _Any:
    def __getattr__(self, name): return _Any()
    def __call__(self, *a, **k): return _Any()
def __getattr__(name): return _Any()
