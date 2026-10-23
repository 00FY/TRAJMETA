"""Persistent isolated CV worker with a hard wall-time deadline per trial."""
import multiprocessing as mp
import warnings


def _worker(connection, X, y, folds, seed):
    from sklearn.model_selection import cross_val_score
    from threadpoolctl import threadpool_limits
    from .automl import build_pipeline
    try:
        while True:
            candidate = connection.recv()
            if candidate is None:
                break
            try:
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter("always")
                    scores = cross_val_score(build_pipeline(candidate, seed), X, y, cv=folds,
                                             scoring="f1_macro", error_score="raise", n_jobs=1)
                connection.send({"scores": scores.tolist(), "warnings": sorted({str(w.message) for w in caught}), "error": None})
            except Exception as exc:
                connection.send({"scores": [], "warnings": [], "error": f"{type(exc).__name__}: {exc}",
                                 "status": "resource_failed" if isinstance(exc, MemoryError) else "failed"})
    finally:
        connection.close()


class CVWorker:
    def __init__(self, X, y, folds, seed, timeout):
        if timeout <= 0:
            raise ValueError("Trial timeout must be positive")
        self.arguments = (X, y, folds, seed)
        self.timeout, self.process, self.connection = timeout, None, None

    def evaluate(self, candidate):
        if self.process is None:
            context = mp.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(target=_worker, args=(child, *self.arguments), daemon=True)
            self.process.start()
            child.close()
        try:
            self.connection.send(candidate)
            if not self.connection.poll(self.timeout):
                self.close()
                return {"scores": [], "warnings": [], "error": "Trial wall-time limit exceeded", "status": "timeout"}
            return self.connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            return {"scores": [], "warnings": [], "error": f"Worker exited: {exc}", "status": "resource_failed"}

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.process.close()
            self.connection.close()
            self.process = self.connection = None
