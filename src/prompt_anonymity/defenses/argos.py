"""Round-trip-translation defense backed by Argos Translate (offline, CPU-friendly).

Scrambles per-user stylometric signal by translating each user turn through a language chain and
back to English: EN -> ZH -> JA -> EN. Argos has no direct zh->ja model, so that hop auto-pivots
through English. The featurize stage re-derives features from the translated text, so this defense
only supplies the rewrite.

Models are downloaded once on first use and are fully offline thereafter.
"""

from __future__ import annotations

from ._backends import PerTurnBatchRewriteDefense


class _ArgosBackend:
    """argos-translate round-trip translator. ``HOPS`` is the logical chain; ``REQUIRED`` is the
    set of *direct* packages that must be installed for every hop to resolve (zh->ja needs
    zh->en + en->ja because Argos pivots that hop through English)."""

    HOPS = [("en", "zh"), ("zh", "ja"), ("ja", "en")]
    REQUIRED = [("en", "zh"), ("zh", "en"), ("en", "ja"), ("ja", "en")]

    def __init__(self):
        # Imported lazily so importing this module never requires argostranslate.
        import argostranslate.package as package

        package.update_package_index()
        available = package.get_available_packages()
        installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}

        # Download+install any missing direct package once; later runs are fully offline.
        for from_code, to_code in (pair for pair in self.REQUIRED if pair not in installed):
            match = next(
                (p for p in available if p.from_code == from_code and p.to_code == to_code),
                None,
            )
            if match is None:
                raise RuntimeError(f"No argos package available for {from_code}->{to_code}")
            package.install_from_path(match.download())

    def roundtrip(self, text: str) -> str:
        import argostranslate.translate as translate

        for from_code, to_code in self.HOPS:  # zh->ja auto-pivots through English
            text = translate.translate(text, from_code, to_code)
        return text


class ArgosRTTDefense(PerTurnBatchRewriteDefense):
    """EN->ZH->JA->EN round-trip translation via Argos Translate, applied per user turn.

    Argos is per-item (CPU), so :meth:`rewrite_batch` just loops; the per-turn cache still dedupes
    identical turns across conversations. The translator is built lazily on first use, so a
    fully-cached run loads no models.
    """

    name = "rtt_argos"
    version = "1"

    def __init__(self):
        self._backend = None

    def _get_backend(self) -> _ArgosBackend:
        if self._backend is None:
            self._backend = _ArgosBackend()
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        backend = self._get_backend()
        return [backend.roundtrip(text) for text in texts]
