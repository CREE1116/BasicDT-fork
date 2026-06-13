from __future__ import annotations

import ctypes
import os
import platform
import numpy as np

# ── BasicDT C bindings ───────────────────────────────────────────────────

_lib = None
_EXT_DIR = os.path.join(os.path.dirname(__file__), "_ext")

_pf = ctypes.POINTER(ctypes.c_float)
_pi = ctypes.POINTER(ctypes.c_int)
_pu8 = ctypes.POINTER(ctypes.c_uint8)


def _lib_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libbasicdt.dylib"
    elif system == "Windows":
        return "basicdt.dll"
    return "libbasicdt.so"


def _get_basicdt_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_p = os.path.join(_EXT_DIR, _lib_filename())
    if not os.path.exists(lib_p):
        raise FileNotFoundError(
            f"[basicdt] Compiled BasicDT C++ binary not found at {lib_p}.\n"
            "Please build/install the package first."
        )

    lib = ctypes.CDLL(lib_p)

    # ── tree structure (de)serialization ─────────────────────────────────
    lib.basicdt_get_K.argtypes = [ctypes.c_void_p]
    lib.basicdt_get_K.restype = ctypes.c_int
    lib.basicdt_get_max_depth.argtypes = [ctypes.c_void_p]
    lib.basicdt_get_max_depth.restype = ctypes.c_int
    lib.basicdt_get_total_nodes.argtypes = [ctypes.c_void_p]
    lib.basicdt_get_total_nodes.restype = ctypes.c_int
    lib.basicdt_get_D.argtypes = [ctypes.c_void_p]
    lib.basicdt_get_D.restype = ctypes.c_int

    lib.basicdt_export.argtypes = [ctypes.c_void_p, _pi, _pf, _pf, _pu8, _pi, _pi, _pu8]
    lib.basicdt_export.restype = None

    lib.basicdt_export_gain.argtypes = [ctypes.c_void_p, _pf]
    lib.basicdt_export_gain.restype = None

    lib.basicdt_from_arrays.argtypes = [
        _pi, _pf, _pf, _pu8, _pi, _pi, _pu8,
        ctypes.c_int,  # total_nodes
        ctypes.c_int,  # K
        ctypes.c_int,  # max_depth
        ctypes.c_int,  # D
    ]
    lib.basicdt_from_arrays.restype = ctypes.c_void_p

    lib.basicdt_ctx_create.argtypes = [
        _pf,           # X     [N, D]
        ctypes.c_int,  # N
        ctypes.c_int,  # D
        ctypes.c_int,  # D_num
        _pi,           # sub   [Ns]
        ctypes.c_int,  # Ns
        ctypes.c_int,  # max_bin
    ]
    lib.basicdt_ctx_create.restype = ctypes.c_void_p

    lib.basicdt_ctx_free.argtypes = [ctypes.c_void_p]
    lib.basicdt_ctx_free.restype = None

    lib.basicdt_set_num_threads.argtypes = [ctypes.c_int]
    lib.basicdt_set_num_threads.restype = None

    lib.basicdt_build.argtypes = [
        ctypes.c_void_p,  # ctx
        _pf,              # G        [N, K]
        _pf,              # H        [N, K]
        ctypes.c_int,     # K
        _pi,              # sub      [Ns]
        ctypes.c_int,     # Ns
        ctypes.c_int,     # max_depth
        ctypes.c_int,     # max_leaves
        ctypes.c_float,   # reg_lambda
        ctypes.c_float,   # colsample (bynode)
        ctypes.c_uint,    # col_seed
        ctypes.c_float,   # gamma
        ctypes.c_float,   # min_child_weight
        ctypes.c_float,   # reg_alpha
        _pf,              # out_pred [N, K]
    ]
    lib.basicdt_build.restype = ctypes.c_void_p

    lib.basicdt_predict.argtypes = [
        ctypes.c_void_p,  # tree_handle
        _pf,              # X     [N, D]  raw
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        _pf,              # out_pred [N, K]
    ]
    lib.basicdt_predict.restype = None

    lib.basicdt_predict_ensemble.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # tree handles [n_trees]
        ctypes.c_int,                     # n_trees
        _pf,                              # X        [N, D] raw
        ctypes.c_int,                     # N
        ctypes.c_int,                     # K
        ctypes.c_float,                   # lr
        _pf,                              # out_pred [N, K]
    ]
    lib.basicdt_predict_ensemble.restype = None

    lib.basicdt_tree_free.argtypes = [ctypes.c_void_p]
    lib.basicdt_tree_free.restype = None

    lib.basicdt_tree_meta_sizes.argtypes = [ctypes.c_void_p, _pi]
    lib.basicdt_tree_meta_sizes.restype = None

    lib.basicdt_tree_export_meta.argtypes = [ctypes.c_void_p, _pf, _pi, _pi, _pf]
    lib.basicdt_tree_export_meta.restype = None

    lib.basicdt_tree_import_meta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, _pf, ctypes.c_int,
        _pi, ctypes.c_int, _pi, _pf,
    ]
    lib.basicdt_tree_import_meta.restype = None

    lib.basicdt_update_gradients.argtypes = [
        _pf,           # F
        _pf,           # oh
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        _pf,           # G
        _pf,           # H
    ]
    lib.basicdt_update_gradients.restype = None

    _lib = lib
    return lib


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(_pf)


def _iptr(a: np.ndarray):
    return a.ctypes.data_as(_pi)


def set_num_threads(n: int) -> None:
    """Set the OpenMP team size for builds/predicts (n <= 0 = all cores)."""
    _get_basicdt_lib().basicdt_set_num_threads(int(n))


def update_gradients(F: np.ndarray, oh: np.ndarray, G: np.ndarray, H: np.ndarray) -> None:
    N, K = F.shape
    lib = _get_basicdt_lib()
    lib.basicdt_update_gradients(
        _fptr(F),
        _fptr(oh),
        N,
        K,
        _fptr(G),
        _fptr(H),
    )


def predict_ensemble(trees: list, X: np.ndarray, K: int, lr: float,
                      F_init: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    N = X.shape[0]
    out = np.tile(np.asarray(F_init, dtype=np.float32), (N, 1))
    handles = [t._tree_handle for t in trees if t._tree_handle is not None]
    if not handles:
        return out
    arr = (ctypes.c_void_p * len(handles))(*handles)
    lib = _get_basicdt_lib()
    lib.basicdt_predict_ensemble(arr, len(handles), _fptr(X), N, K,
                                 ctypes.c_float(lr), _fptr(out))
    return out


class BasicDTree:
    def __init__(
        self,
        max_depth:    int   = 4,
        max_leaves:   int | None = None,
        reg_lambda:   float = 1.0,
        subsample:    float = 1.0,
        random_state: int | None = None,
        gamma:        float = 0.0,
        min_child_weight: float = 1.0,
        reg_alpha:    float = 0.0,
    ):
        self.max_depth    = max_depth
        self.max_leaves   = max_leaves
        self.reg_lambda   = reg_lambda
        self.subsample    = subsample
        self.random_state = random_state
        self.gamma        = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha    = reg_alpha
        self._tree_handle = None
        self._K           = None

    @classmethod
    def _from_handle(cls, handle, K: int, max_depth: int, reg_lambda: float, max_leaves: int | None = None,
                     gamma: float = 0.0, min_child_weight: float = 1.0, reg_alpha: float = 0.0):
        t = cls(max_depth=max_depth, max_leaves=max_leaves, reg_lambda=reg_lambda,
                gamma=gamma, min_child_weight=min_child_weight, reg_alpha=reg_alpha)
        t._tree_handle = handle
        t._K = K
        return t

    def fit_predict(
        self,
        X:      np.ndarray,
        G:      np.ndarray,
        H:      np.ndarray,
        D_num:  int | None = None,
        subset: np.ndarray | None = None,
    ) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        G = np.ascontiguousarray(G, dtype=np.float32)
        H = np.ascontiguousarray(H, dtype=np.float32)
        N, _ = X.shape
        K = G.shape[1]

        if subset is None:
            if self.subsample < 1.0:
                rng    = np.random.default_rng(self.random_state)
                ns     = max(1, int(N * self.subsample))
                subset = rng.choice(N, ns, replace=False).astype(np.int32)
            else:
                subset = np.arange(N, dtype=np.int32)

        max_l = self.max_leaves
        if max_l is None or max_l <= 0:
            max_l = 1 << self.max_depth
        ctx = BasicDContext(X, D_num=D_num)
        try:
            tree, out_pred = ctx.build(
                G, H, subset, self.max_depth, max_l, self.reg_lambda,
                gamma=self.gamma, min_child_weight=self.min_child_weight,
                reg_alpha=self.reg_alpha,
            )
        finally:
            ctx.close()
        self._tree_handle = tree._tree_handle
        tree._tree_handle = None
        self._K = K
        return out_pred

    def predict(self, X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if self._tree_handle is None:
            raise RuntimeError("Tree is not fitted.")
        X   = np.ascontiguousarray(X, dtype=np.float32)
        N   = X.shape[0]
        if out is None:
            out = np.zeros((N, self._K), dtype=np.float32)
        else:
            if out.shape != (N, self._K) or out.dtype != np.float32 or not out.flags.c_contiguous:
                raise ValueError("out must be a contiguous float32 array of shape (N, K)")
        lib = _get_basicdt_lib()
        lib.basicdt_predict(self._tree_handle, _fptr(X), N, self._K, _fptr(out))
        return out

    def export_arrays(self) -> dict:
        """Return the fitted tree as flat numpy arrays for inspection/explanation.

        Keys: split_feature, threshold, leaf_vals (n_nodes×K), is_leaf,
        left_child, right_child, split_gain, n_nodes, K. Internal nodes hold
        split_feature/threshold/gain; leaves hold the K-vector leaf_vals.
        """
        if self._tree_handle is None:
            raise RuntimeError("Tree is not fitted.")
        lib     = _get_basicdt_lib()
        h       = self._tree_handle
        n_nodes = lib.basicdt_get_total_nodes(h)
        K       = self._K

        split_feature = np.empty(n_nodes,     dtype=np.int32)
        threshold     = np.empty(n_nodes,     dtype=np.float32)
        leaf_vals     = np.empty(n_nodes * K, dtype=np.float32)
        is_leaf       = np.empty(n_nodes,     dtype=np.uint8)
        left_child    = np.empty(n_nodes,     dtype=np.int32)
        right_child   = np.empty(n_nodes,     dtype=np.int32)
        default_left  = np.empty(n_nodes,     dtype=np.uint8)
        split_gain    = np.empty(n_nodes,     dtype=np.float32)
        lib.basicdt_export(
            h, _iptr(split_feature), _fptr(threshold), _fptr(leaf_vals),
            is_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            _iptr(left_child), _iptr(right_child),
            default_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        lib.basicdt_export_gain(h, _fptr(split_gain))
        return {
            "split_feature": split_feature,
            "threshold":     threshold,
            "leaf_vals":     leaf_vals.reshape(n_nodes, K),
            "is_leaf":       is_leaf,
            "left_child":    left_child,
            "right_child":   right_child,
            "default_left":  default_left,
            "split_gain":    split_gain,
            "n_nodes":       n_nodes,
            "K":             K,
        }

    def __getstate__(self):
        base = {
            "max_depth":    self.max_depth,
            "reg_lambda":   self.reg_lambda,
            "subsample":    self.subsample,
            "random_state": self.random_state,
            "K":            self._K,
            "handle":       None,
        }
        if self._tree_handle is None:
            return base
        lib     = _get_basicdt_lib()
        h       = self._tree_handle
        n_nodes = lib.basicdt_get_total_nodes(h)
        K       = self._K
        D       = lib.basicdt_get_D(h)

        split_feature   = np.empty(n_nodes,     dtype=np.int32)
        threshold = np.empty(n_nodes,     dtype=np.float32)
        leaf_vals = np.empty(n_nodes * K, dtype=np.float32)
        is_leaf   = np.empty(n_nodes,     dtype=np.uint8)
        left_child = np.empty(n_nodes,     dtype=np.int32)
        right_child = np.empty(n_nodes,     dtype=np.int32)
        default_left = np.empty(n_nodes,     dtype=np.uint8)
        lib.basicdt_export(
            h, _iptr(split_feature), _fptr(threshold), _fptr(leaf_vals),
            is_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            _iptr(left_child), _iptr(right_child),
            default_left.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )

        sizes = np.zeros(4, dtype=np.int32)
        lib.basicdt_tree_meta_sizes(h, _iptr(sizes))
        D_num, D_cat, n_entries, na_len = (int(v) for v in sizes)
        na_means  = np.zeros(max(na_len, 1),    dtype=np.float32)
        cat_sizes = np.zeros(max(D_cat, 1),     dtype=np.int32)
        cat_keys  = np.zeros(max(n_entries, 1), dtype=np.int32)
        cat_vals  = np.zeros(max(n_entries, 1), dtype=np.float32)
        lib.basicdt_tree_export_meta(
            h, _fptr(na_means), _iptr(cat_sizes), _iptr(cat_keys),
            _fptr(cat_vals),
        )

        base.update({
            "handle":        "serialized",
            "tree_max_depth": lib.basicdt_get_max_depth(h),
            "n_nodes":       n_nodes,
            "D":             D,
            "split_feature": split_feature,
            "threshold":     threshold,
            "leaf_vals":     leaf_vals,
            "is_leaf":       is_leaf,
            "left_child":    left_child,
            "right_child":   right_child,
            "default_left":  default_left,
            "D_num":         D_num,
            "D_cat":         D_cat,
            "na_len":        na_len,
            "na_means":      na_means[:na_len],
            "cat_sizes":     cat_sizes[:D_cat],
            "cat_keys":      cat_keys[:n_entries],
            "cat_vals":      cat_vals[:n_entries],
            "max_leaves":    self.max_leaves,
            "gamma":         self.gamma,
            "min_child_weight": self.min_child_weight,
            "reg_alpha":     self.reg_alpha,
        })
        return base

    def __setstate__(self, state):
        self.max_depth    = state["max_depth"]
        self.max_leaves   = state.get("max_leaves")
        self.reg_lambda   = state["reg_lambda"]
        self.subsample    = state.get("subsample", 1.0)
        self.random_state = state.get("random_state")
        self.gamma        = state.get("gamma", 0.0)
        self.min_child_weight = state.get("min_child_weight", 1.0)
        self.reg_alpha    = state.get("reg_alpha", 0.0)
        self._K           = state["K"]
        self._tree_handle = None
        if state["handle"] is None:
            return
        lib = _get_basicdt_lib()
        s   = state
        default_l = s.get("default_left", None)
        if default_l is None:
            default_l = np.ones(s["n_nodes"], dtype=np.uint8)
        else:
            default_l = np.ascontiguousarray(default_l, dtype=np.uint8)
        handle = lib.basicdt_from_arrays(
            _iptr(s["split_feature"]), _fptr(s["threshold"]), _fptr(s["leaf_vals"]),
            s["is_leaf"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            _iptr(s["left_child"]), _iptr(s["right_child"]),
            default_l.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            s["n_nodes"], s["K"], s["tree_max_depth"], s["D"]
        )
        na_means  = np.ascontiguousarray(s["na_means"],  dtype=np.float32)
        cat_sizes = np.ascontiguousarray(s["cat_sizes"], dtype=np.int32)
        cat_keys  = np.ascontiguousarray(s["cat_keys"],  dtype=np.int32)
        cat_vals  = np.ascontiguousarray(s["cat_vals"],  dtype=np.float32)
        lib.basicdt_tree_import_meta(
            handle, s["D_num"], _fptr(na_means), s["na_len"],
            _iptr(cat_sizes), s["D_cat"], _iptr(cat_keys), _fptr(cat_vals),
        )
        self._tree_handle = handle

    def __del__(self):
        if self._tree_handle is not None:
            try:
                _get_basicdt_lib().basicdt_tree_free(self._tree_handle)
            except Exception:
                pass
            self._tree_handle = None


class BasicDContext:
    def __init__(self, X: np.ndarray, D_num: int | None = None, max_bin: int = 256):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.N, self.D = self.X.shape
        self.D_num = self.D if D_num is None else int(D_num)
        self.max_bin = int(max_bin)
        sub = np.arange(self.N, dtype=np.int32)
        lib = _get_basicdt_lib()
        self._handle = lib.basicdt_ctx_create(
            _fptr(self.X), self.N, self.D, self.D_num, _iptr(sub), self.N,
            self.max_bin,
        )

    def build(
        self,
        G: np.ndarray,
        H: np.ndarray,
        sub: np.ndarray,
        max_depth: int,
        max_leaves: int,
        reg_lambda: float,
        colsample: float = 1.0,
        col_seed: int = 0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        out_pred: np.ndarray | None = None,
    ) -> tuple[BasicDTree, np.ndarray]:
        if self._handle is None:
            raise RuntimeError("Context is closed.")
        G = np.ascontiguousarray(G, dtype=np.float32)
        H = np.ascontiguousarray(H, dtype=np.float32)
        sub = np.ascontiguousarray(sub, dtype=np.int32)
        K = G.shape[1]
        if out_pred is None:
            out_pred = np.zeros((self.N, K), dtype=np.float32)
        else:
            if out_pred.shape != (self.N, K) or out_pred.dtype != np.float32 or not out_pred.flags.c_contiguous:
                raise ValueError("out_pred must be a contiguous float32 array of shape (N, K)")
        lib = _get_basicdt_lib()
        handle = lib.basicdt_build(
            self._handle, _fptr(G), _fptr(H), K,
            _iptr(sub), len(sub), max_depth, max_leaves,
            ctypes.c_float(reg_lambda),
            ctypes.c_float(colsample),
            ctypes.c_uint(col_seed & 0xFFFFFFFF),
            ctypes.c_float(gamma),
            ctypes.c_float(min_child_weight),
            ctypes.c_float(reg_alpha),
            _fptr(out_pred),
        )

        tree = BasicDTree._from_handle(handle, K, max_depth, reg_lambda, max_leaves=max_leaves,
                                       gamma=gamma, min_child_weight=min_child_weight, reg_alpha=reg_alpha)
        return tree, out_pred

    def close(self):
        if self._handle is not None:
            try:
                _get_basicdt_lib().basicdt_ctx_free(self._handle)
            except Exception:
                pass
            self._handle = None

    def __del__(self):
        self.close()
