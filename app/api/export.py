"""CSV export endpoint.

Produces a standard CSV file of every buffered reading so that the data can
be analysed offline in Excel, pandas or any BI tool.
"""

from __future__ import annotations

import csv
import io
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_service
from app.services.machine_service import MachineService

router = APIRouter(prefix="/api", tags=["export"])

#: Column order of the exported CSV file.
CSV_COLUMNS: list[str] = [
    "timestamp",
    "machine_id",
    "machine_type",
    "temperature",
    "vibration",
    "current",
    "rpm",
    "pressure",
]


@router.get("/export/csv")
async def export_csv(
    service: Annotated[MachineService, Depends(get_service)],
    machine_id: Annotated[
        str | None, Query(description="Restrict the export to one machine")
    ] = None,
    limit: Annotated[
        int | None, Query(ge=1, le=10000, description="Maximum rows to export")
    ] = None,
) -> StreamingResponse:
    """Export the buffered readings as a downloadable CSV file.

    Args:
        machine_id: Optional filter to a single machine.
        limit: Optional cap on the number of rows.
        service: The injected machine service.

    Returns:
        A streaming CSV response with a ``Content-Disposition`` attachment
        header so browsers download the file rather than display it.
    """
    if machine_id:
        rows = service.get_readings(machine_id, limit or 10000)
    else:
        rows = service.export_rows()
        if limit:
            rows = rows[-limit:]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for reading in rows:
        writer.writerow(
            {
                "timestamp": reading.timestamp.isoformat(),
                "machine_id": reading.machine_id,
                "machine_type": reading.machine_type.value,
                "temperature": reading.temperature,
                "vibration": reading.vibration,
                "current": reading.current,
                "rpm": reading.rpm,
                "pressure": reading.pressure,
            }
        )
    buffer.seek(0)

    filename = f"readings_{machine_id or 'all'}.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
