"""
"""

from __future__ import annotations

import re
import argparse
from datetime import datetime
from collections import namedtuple


try:
    from polly.kpf import\
        ORDERLETS, TIMESOFDAY, LFC_ORDER_INDICES, THORIUM_ORDER_INDICES
except ImportError:
    from kpf import\
        ORDERLETS, TIMESOFDAY, LFC_ORDER_INDICES, THORIUM_ORDER_INDICES



Mask = namedtuple("Mask", ["date", "timeofday", "orderlet"])


def parse_date_string(datestr: str) -> datetime:
    
    # Handle dates like "2024-12-31"
    if "-" in datestr:
        datestr = "".join(datestr.split("-"))
    # Now it should be "20241231"

    year = int(datestr[:4])
    month = int(datestr[4:6])
    day = int(datestr[6:])
    
    return datetime(year=year, month=month, day=day)


def parse_filename(filename: str | list[str]) -> tuple[datetime, str, str]:
    
    if isinstance(filename, list):
        return [parse_filename(f) for f in filename]
    
    filename = filename.split("/")[-1]
    datestr, timeofday, orderlet, *_ = filename.split("_")[:3]
    date = parse_date_string(datestr)
    
    return Mask(date=date, timeofday=timeofday, orderlet=orderlet)


def parse_yyyymmdd(input: str | float | int) -> datetime:
    
    if input == "now":
        return datetime.now()
    
    if isinstance(input, float):
        input = str(int(input))
    elif isinstance(input, int):
        input = str(input)
        
    assert isinstance(input, str) and len(input) == 8
    
    yyyy = int(input[0:4])
    mm   = int(input[4:6])
    dd   = int(input[6:8])
    
    return datetime(year = yyyy, month = mm, day = dd)



def parse_num_list(string_list: str) -> list[int]:
    """
    Adapted from Julian Stürmer's PyEchelle code
    
    Converts a string specifying a range of numbers (e.g. '1-3') into a list of
    these numbers ([1,2,3])
    """

    m = re.match(r"(\d+)(?:-(\d+))?$", string_list)
    if not m:
        raise argparse.ArgumentTypeError(
            f"'{string_list}' is not a range or number."+\
            f"Expected forms like '1-12' or '6'."
            )
    
    start = m.group(1)
    end = m.group(2) or start
    
    return list(range(int(start), int(end) + 1))



def parse_orders(orders_str: str) -> list[int]:
    """
    Wrapper around parse_num_list
    """
    
    if (orders_str == "all") or (orders_str is None):
        return list(range((67)))
    elif orders_str == "lfc":
        return LFC_ORDER_INDICES
    elif orders_str == "thorium":
        return THORIUM_ORDER_INDICES
        
    else:
        try:
            return parse_num_list(orders_str)
        except Exception as e:
            print(f"Exception raised when parsing orders: {e}")
            print("Returning ALL orders")
            return list(range((67)))



def parse_timesofday(timesofday: str) -> list:
    if (timesofday == "all") or (timesofday is None):
        return TIMESOFDAY
    
    elif "," in timesofday:
        return timesofday.split(sep=",")
    
    else:
        return [timesofday]


def parse_orderlets(orderlets: str) -> list:
    
    if (orderlets == "all") or (orderlets is None):
        return ORDERLETS
    
    elif "," in orderlets:
        orderlets = orderlets.split(sep=",")
        for ol in orderlets:
            assert ol in ORDERLETS
        return orderlets
    
    else:
        assert orderlets in ORDERLETS
        return [orderlets]
    
    
def parse_bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif string.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
    
def get_orderlet_name(orderlet: str) -> str:
    """
    A simple helper function to get the non-numeric part of the orderlet name,
    used to build the relevant FITS header keyword to access data.
    
    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return 'SCI'
    """
    
    if orderlet.startswith("SCI"):
        return "SCI"
    else:
        return orderlet
    
    
def get_orderlet_index(orderlet: str) -> str:
    """
    A simple helper function to get only the numeric part of the orderlet name,
    used to build the relevant FITS header keyword to access data.
    
    eg. for 'SCI1' we need 'GREEN_SCI_FLUX1', so return '1'
    """
    
    if orderlet.startswith("SCI"):
        return orderlet[-1]
    else:
        return ""