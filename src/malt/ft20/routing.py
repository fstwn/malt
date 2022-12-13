# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

from typing import Tuple


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import requests
import json


# API SETTINGS ----------------------------------------------------------------

_ROUTING_URL = 'http://router.project-osrm.org/route/v1/car/{0},{1};{2},{3}{4}'


# FUNCTION DEFINITIONS --------------------------------------------------------

def get_distance_latlon(lat_a: float,
                        lon_a: float,
                        lat_b: float,
                        lon_b: float,
                        url: str = _ROUTING_URL):
    """
    Queries the OSRM routing server for a distance between the two locations.
    Returns distance in kilometers
    """
    options = '?overview=false'
    query = url.format(lon_a, lat_a, lon_b, lat_b, options)
    routes = json.loads(requests.get(query).content)
    distance = routes.get('routes')[0]['distance']
    return distance * 0.001


def get_distance(loc_a: Tuple[float, float],
                 loc_b: Tuple[float, float],
                 url: str = _ROUTING_URL):
    """
    Queries the OSRM routing server for a distance between the two locations.
    Returns distance in kilometers.
    NOTE: Location tuples are defined as (lat, lon)!
    """
    lat_a, lon_a = loc_a
    lat_b, lon_b = loc_b
    return get_distance_latlon(lat_a, lon_a, lat_b, lon_b, url=url)


# RUN SCRIPT ------------------------------------------------------------------

if __name__ == "__main__":
    loc_a = (49.863323, 8.678350)
    loc_b = (52.284726, 10.548717)
    print(f'Distance is {get_distance(loc_a, loc_b)} km.')
