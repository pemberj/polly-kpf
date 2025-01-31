"""
polly

fileselection

Contains functions that search the shrek filesystem for relevant files, either
KPF L1 FITS files or mask/drift files generated by Polly
"""

# from __future__ import annotations

from datetime import datetime
from pathlib import Path

from astropy.io import fits

from polly.kpf import L1_DIR, MASTERS_DIR, ORDERLETS, TIMESOFDAY
from polly.log import logger
from polly.parsing import parse_filename, parse_yyyymmdd


def find_L1_etalon_files(
    date: str,
    timeofday: str,
    masters: bool,
    pp: str = "",
) -> str | list[str]:
    """
    Locates relevant L1 files for a given date and time of day. At the moment
    it loops through all files and looks at the "OBJECT" keyword in their
    headers.

    TODO: Don't just take every matching frame! There are three "blocks" of
          three etalon frames taken every morning (and evening?). Should take
          only the single block that is closest to the SoCal observations.
    TODO: Use a database lookup (on shrek) to select files?
    """

    assert timeofday in TIMESOFDAY

    if masters:
        p = MASTERS_DIR / f"{date}"

        files = p.glob(
            f"kpf_{date}_master_arclamp_autocal-etalon-all-{timeofday}_L1.fits"
        )
        try:
            assert len(files) == 1
        except AssertionError:
            logger.info(f"{pp}{len(files)} files found")
            return None

        f_p = Path(files[0])

        with f_p.open(mode="rb") as f:
            try:
                _object = fits.getval(f, "OBJECT")
                if "etalon" in _object.lower():
                    return files[0]
            except (FileNotFoundError, OSError) as e:
                logger.error(f"{pp}{e}")
                return None

    p: Path = L1_DIR / f"{date}"

    all_files: list[str] = p.glob("*.fits")

    out_files: list[str] = []

    for f in all_files:
        try:
            _object = fits.getval(f, "OBJECT")
        except (FileNotFoundError, OSError) as e:
            logger.error(f"{pp}{e}")
            continue

        if "etalon" in _object.lower():
            file_timeofday = _object.split("-")[-1]
            if file_timeofday == timeofday:
                out_files.append(f)

    return out_files


def find_mask(
    masks: list[str],
    datestr: str | None = None,
    date: datetime | None = None,
    timeofday: str | list[str] | None = None,
    orderlet: str | list[str] | None = None,
) -> str:
    """
    Find a single mask matching the input criteria. To be used to locate the
    reference mask for a drift analysis.
    """

    if timeofday is None:
        timesofday = TIMESOFDAY
    elif isinstance(timeofday, str):
        assert timeofday in TIMESOFDAY
        timesofday = [timeofday]
    elif isinstance(timeofday, list):
        for t in timeofday:
            assert t in TIMESOFDAY
        timesofday = timeofday

    if orderlet is None:
        orderlets = ORDERLETS
    elif isinstance(orderlet, str):
        assert orderlet in ORDERLETS
        orderlets = [orderlet]

    if (datestr is None and date is None) or (datestr and date):
        print("Exactly one of `datestr` or `date` must be specified")

    if date:
        try:
            assert isinstance(date, datetime)
        except AssertionError:
            print(date)

    if datestr:
        date = parse_yyyymmdd(datestr)

    for m in masks:
        mdate, mtimeofday, morderlet = parse_filename(m)
        if (mdate == date) and (mtimeofday in timesofday) and (morderlet in orderlets):
            return m

    return ""


def select_masks(
    masks: list[str],
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    timeofday: str | list[str] | None = None,
    orderlet: str | list[str] | None = None,
) -> list[str]:
    """
    Find all masks in an input list that match the given criteria.
    """

    if isinstance(orderlet, str):
        orderlet = [orderlet]
    for ol in orderlet:
        assert ol in [*ORDERLETS, None]

    if isinstance(timeofday, str):
        timeofday = [timeofday]
    for tod in timeofday:
        assert tod in [*TIMESOFDAY, None]

    # Start with the full list of masks
    valid_masks = masks

    # Then progressively keep only the matching masks
    if min_date:
        valid_masks = [m for m in valid_masks if parse_filename(m).date >= min_date]

    if max_date:
        valid_masks = [m for m in valid_masks if parse_filename(m).date <= max_date]

    if timeofday:
        valid_masks = [
            m for m in valid_masks if parse_filename(m).timeofday in timeofday
        ]

    if orderlet:
        valid_masks = [m for m in valid_masks if parse_filename(m).orderlet in orderlet]

    # Only the matching masks are left
    if not valid_masks:
        print("No matching masks found!")
        return None

    return valid_masks
