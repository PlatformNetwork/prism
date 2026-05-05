from __future__ import annotations

from datetime import UTC, datetime
from typing import SupportsFloat, cast

from .repository import PrismRepository, epoch_id_for


async def get_weights(repository: PrismRepository, epoch_seconds: int) -> dict[str, float]:
    epoch_id = epoch_id_for(datetime.now(UTC), epoch_seconds)
    rows = await repository.score_rows(epoch_id)
    best: dict[str, float] = {}
    for row in rows:
        hotkey = str(row["hotkey"])
        score = max(0.0, float(cast(SupportsFloat, row["final_score"])))
        best[hotkey] = max(best.get(hotkey, 0.0), score)
    total = sum(best.values())
    if total <= 0:
        return {}
    return {hotkey: score / total for hotkey, score in best.items() if score > 0}
