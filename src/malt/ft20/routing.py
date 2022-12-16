# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import math
import time
from typing import List, Sequence, Tuple


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import requests
import json


# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt.ft20.RepositoryComponent import RepositoryComponent


# API SETTINGS ----------------------------------------------------------------

_ROUTING_URL = 'http://router.project-osrm.org/route/v1/car/{0},{1};{2},{3}{4}'
_OVERPASS_URL = 'http://overpass-api.de/api/interpreter'
_DBFL_LOCATION = ()
_LAST_REQUEST = 0


# FUNCTION DEFINITIONS --------------------------------------------------------

def get_concrete_landfills():
    """
    QUery the overpass API for all concrete landfills in Germany.
    """
    query = (
        '[out:json];'
        'area["ISO3166-1"="DE"][admin_level=2];'
        '('
        'nwr["landuse"="landfill"]["name"~"Bauschutt"](area);'
        'nwr["landuse"="landfill"]["name"~"Beton"](area);'
        ');'
        'out center;'
        )
    global _OVERPASS_URL
    response = requests.get(_OVERPASS_URL,
                            params={'data': query})
    data = response.json()
    return _sanitize_overpass_data(data)


def get_concrete_factories():
    query = (
        '[out:json];'
        'area["ISO3166-1"="DE"][admin_level=2];'
        '('
        'nwr["landuse"="industrial"]["name"~"Betonwerk"](area);'
        'nwr["landuse"="industrial"]["name"~"Betonmischwerk"](area);'
        ');'
        'out center;'
        )
    global _OVERPASS_URL
    response = requests.get(_OVERPASS_URL,
                            params={'data': query})
    data = response.json()
    return _sanitize_overpass_data(data)


def query_distance_latlon(lat_a: float,
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
    global _LAST_REQUEST
    _LAST_REQUEST = time.time()
    distance = routes.get('routes')[0]['distance']
    return distance * 0.001


def query_distance(loc_a: Tuple[float, float],
                   loc_b: Tuple[float, float],
                   url: str = _ROUTING_URL):
    """
    Queries the OSRM routing server for a distance between the two locations.
    Returns distance in kilometers.
    NOTE: Location tuples are defined as (lat, lon)!
    """
    lat_a, lon_a = loc_a
    lat_b, lon_b = loc_b
    return query_distance_latlon(lat_a, lon_a, lat_b, lon_b, url=url)


def get_distance(loc_a: Tuple[float, float],
                 loc_b: Tuple[float, float],
                 url: str = _ROUTING_URL):
    """
    Queries the OSRM routing server for a distance between the two locations,
    respecting the 1 request per second rule.

    Returns distance in kilometers.
    NOTE: Location tuples are defined as (lat, lon)!
    """
    global _LAST_REQUEST
    if time.time() - 1.1 >= _LAST_REQUEST:
        return query_distance(loc_a, loc_b, url=url)
    else:
        time.sleep(1.0 - (time.time() - _LAST_REQUEST))
        return query_distance(loc_a, loc_b, url=url)


def compute_transport_distances(
        repository_components: List[RepositoryComponent],
        site_location: Tuple[float, float]
        ):
    """
    Computes the transport distances between repository components and a
    specific site (target) location in a way that limits the amount of
    requests to the OSRM API to save time and calculation time.
    """
    uloc = {}
    distances = []
    # loop over repository components and create dict of unique locations
    for i, comp in enumerate(repository_components):
        loc = comp.location
        strloc = _loc_to_str(loc)
        # test if key is already present
        if strloc not in uloc:
            dist = get_distance(loc, site_location)
            uloc[strloc] = dist
            distances.append(dist)
        else:
            distances.append(uloc[strloc])
    return distances


def calculate_transporthistory(transporthistory: Sequence[Sequence]):
    known_routes = {}
    for i, transport in enumerate(transporthistory):
        if int(transport[2]) == -1:
            loc_a = transport[0]
            loc_b = transport[1]
            loc_a_str = _loc_to_str(loc_a)
            loc_b_str = _loc_to_str(loc_b)
            route_a = loc_a_str + '->' + loc_b_str
            route_b = loc_b_str + '->' + loc_a_str
            if route_a not in known_routes:
                dist = get_distance(loc_a, loc_b)
                known_routes[route_a] = dist
                known_routes[route_b] = dist
            transporthistory[i][2] = known_routes[route_a]
    return transporthistory


# UTILITIES -------------------------------------------------------------------

def haversine(origin: Tuple[float, float], destination: Tuple[float, float]):
    """
    Calculate the Haversine distance.
    Source: https://stackoverflow.com/a/57445729

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def _loc_to_str(loc: Tuple[float, float]):
    """
    Utility function to convert a location tuple of signature (lat, lon) to
    a string.
    """
    return str(loc).strip('[]')


def _str_to_loc(locstr: str):
    """
    Utility function to convert a string to a location-tuple with signature
    (lat, lon)
    """
    return tuple([float(c.strip(' ')) for c in locstr[0].split(',')])


def _sanitize_overpass_data(data: dict):
    """
    Sanitizes reponse data from the overpass API.
    Returns a dict in the format {'name' : (lat, lon)}.
    """
    # loop over data and extract relevant information
    facilities = {}
    for element in data['elements']:
        try:
            name = element['tags']['name']
        except KeyError:
            continue
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
        elif 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        facilities[name] = (lat, lon)
    return facilities


# RUN SCRIPT ------------------------------------------------------------------

if __name__ == "__main__":

    # test routing query
    loc_a = (49.863323, 8.678350)
    loc_b = (52.284726, 10.548717)
    print(f'Distance is {get_distance(loc_a, loc_b)} km.')

    # test osm location query
    locs = get_concrete_landfills()
