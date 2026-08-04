from ..topk_bench import Provider

__all__ = ["TopKRsProvider"]


class TopKRsProvider(Provider):
    """Selects the Rust-native client inside the harness.

    Carries no logic: `__native__` tells the Rust side to construct `NativeProvider`
    from the environment instead of calling back into Python. Every other provider is
    driven as a Python object over PyO3, so `topk-py` measures the Python client as much
    as the protocol -- this one takes Python off the hot path entirely, and is the floor
    the others should be read against.
    """

    __native__ = True

    def name(self) -> str:
        return "topk-rs"
