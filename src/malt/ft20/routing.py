# This file implements automatic computation of routes using an OSRM server.
# In this case, the OSRM test server (http://router.project-osrm.org) is used
# under adherence to the API usage policy. We limit requests to 1 req/sec to
# adhere to the policy. The routing service is only used as part of a
# prototype-implementation within the research project 'Fertigteil 2.0'.
#
# Open Data Commons Open Database License (ODbL):
#
# OSRM is made available under the Open Database License:
# http://opendatacommons.org/licenses/odbl/1.0/. Any rights in individual
# contents of the database are licensed under the Database Contents License:
# http://opendatacommons.org/licenses/dbcl/1.0/

# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import datetime
import math
import json
from operator import itemgetter
import os
import time
from typing import List, Sequence, Tuple


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import requests


# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt.ft20.RepositoryComponent import RepositoryComponent
from malt.hopsutilities import sanitize_path


# ENVIRONMENT SETTINGS --------------------------------------------------------

_HERE = os.path.dirname(sanitize_path(__file__))
"""str: Directory of this particular file."""

_LANDFILL_CACHE = sanitize_path(os.path.join(_HERE, 'landfills.json'))
_FACTORY_CACHE = sanitize_path(os.path.join(_HERE, 'factories.json'))

# API SETTINGS ----------------------------------------------------------------

_ROUTING_URL = 'http://router.project-osrm.org/route/v1/car/{0},{1};{2},{3}{4}'
_OVERPASS_URL = 'http://overpass-api.de/api/interpreter'
_LAST_REQUEST = 0
_KNOWN_ROUTES = {}
_USER_AGENT = ('FERTIGTEIL2.0 (prototype application within circular economy '
               'research project at TU Darmstadt, written in Python. Server is'
               'only used for testing purposes.) Please do not ban.')


# FUNCTION DEFINITIONS --------------------------------------------------------

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
    headers = _get_headers()
    routes = json.loads(requests.get(query, headers=headers).content)
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


def get_concrete_landfills():
    """
    Query the overpass API for all concrete landfills in Germany.
    """
    global _OVERPASS_URL
    headers = _get_headers()
    query = (
        '[out:json];'
        'area["ISO3166-1"="DE"][admin_level=2];'
        '('
        'nwr["landuse"="landfill"]["name"~"Bauschutt"](area);'
        'nwr["landuse"="landfill"]["name"~"Beton"](area);'
        ');'
        'out center;'
        )
    response = requests.get(_OVERPASS_URL,
                            params={'data': query},
                            headers=headers)
    data = response.json()
    return _sanitize_overpass_data(data)


def get_concrete_factories():
    """
    Query the overpass API for all concrete factories in Germany.
    """
    global _OVERPASS_URL
    headers = _get_headers()
    query = (
        '[out:json];'
        'area["ISO3166-1"="DE"][admin_level=2];'
        '('
        'nwr["landuse"="industrial"]["name"~"Betonwerk"](area);'
        'nwr["landuse"="industrial"]["name"~"Betonmischwerk"](area);'
        ');'
        'out center;'
        )
    response = requests.get(_OVERPASS_URL,
                            params={'data': query},
                            headers=headers)
    data = response.json()
    return _sanitize_overpass_data(data)


def calculate_transporthistory(transporthistory: Sequence[Sequence]):
    """
    Calculate all missing distances in a transporthistory.
    """
    global _KNOWN_ROUTES
    for i, transport in enumerate(transporthistory):
        if int(transport[2]) == -1:
            loc_a = transport[0]
            loc_b = transport[1]
            loc_a_str = _loc_to_str(loc_a)
            loc_b_str = _loc_to_str(loc_b)
            route_a = loc_a_str + '->' + loc_b_str
            route_b = loc_b_str + '->' + loc_a_str
            if route_a not in _KNOWN_ROUTES:
                dist = get_distance(loc_a, loc_b)
                _KNOWN_ROUTES[route_a] = dist
                _KNOWN_ROUTES[route_b] = dist
            transporthistory[i][2] = _KNOWN_ROUTES[route_a]
    return transporthistory


def compute_transport_to_site(
        repository_components: List[RepositoryComponent],
        site_location: Tuple[float, float]
        ):
    """
    Computes the transport distances between repository components and a
    specific site (target) location in a way that limits the amount of
    requests to the OSRM API to save time and calculation time.
    """
    global _KNOWN_ROUTES
    distances = []
    # loop over repository components and create dict of unique locations
    for i, comp in enumerate(repository_components):
        loc_a = comp.location
        loc_b = site_location
        loc_a_str = _loc_to_str(loc_a)
        loc_b_str = _loc_to_str(loc_b)
        route_a = loc_a_str + '->' + loc_b_str
        route_b = loc_b_str + '->' + loc_a_str
        # test if key is already present
        if route_a not in _KNOWN_ROUTES:
            dist = get_distance(loc_a, loc_b)
            _KNOWN_ROUTES[route_a] = dist
            _KNOWN_ROUTES[route_b] = dist
            distances.append(dist)
        else:
            distances.append(_KNOWN_ROUTES[route_a])
    return distances


def compute_landfill_distances(
        repository_components: List[RepositoryComponent]):
    # retrieve landfill cache filepath from global vars
    global _LANDFILL_CACHE
    # create file version to check against file
    version = int(datetime.date.today().strftime('%Y%m%d'))
    # read from file
    with open(_LANDFILL_CACHE, 'r', encoding='utf-8') as f:
        landfills = json.read(f)
    if landfills['file_version'] < version:
        # get all landfills across germany
        landfills = get_concrete_landfills()
        landfills['file_version'] = version
        with open(_LANDFILL_CACHE, 'w', encoding='utf-8') as f:
            json.dump(landfills, f, ensure_ascii=False, indent=4)
        del landfills['file_version']

    # compute haversine distance for all locations to find nearest landfill
    landfill_distances = []
    for i, rc in enumerate(repository_components):
        loc_a = rc.transportistory_parsed[0][0]
        lfdists = sorted([(k, haversine(loc_a, landfills[k]))
                         for k in landfills.keys()], key=itemgetter(1))
        nlf = lfdists[0]
        # compute car-routed distance to nearest landfill using OSRM
        dist = get_distance(loc_a, nlf[1])
        landfill_distances.append(dist)

    return landfill_distances


def compute_factory_distances(
        repository_components: List[RepositoryComponent]):
    pass


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


def _get_headers(user_agent: str = _USER_AGENT):
    """
    Utility function to retrieve standard headers and attach uaser agent
    string for making API requests.
    """
    headers = requests.utils.default_headers()
    headers.update({'User-Agent': user_agent})
    return headers


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
    loc_a = (50.284726, 10.548717)
    landfills = get_concrete_landfills()
    lfdists = sorted([(k, haversine(loc_a, landfills[k]))
                      for k in landfills.keys()], key=itemgetter(1))
    nlf = lfdists[0]
    print(lfdists)
    print(nlf)
