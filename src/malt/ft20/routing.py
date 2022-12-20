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
"""str: Path to directory of this particular file."""

_LANDFILL_CACHE = sanitize_path(os.path.join(_HERE, 'landfills.json'))
"""str: Path to the file where landfill locations are cached."""

_LANDFILL_CACHING_INTERVAL = 12
"""int: Interval in hours defining the refresh rate of the cache."""

_FACTORY_CACHE = sanitize_path(os.path.join(_HERE, 'factories.json'))
"""str: Path to the file where concrete factories are cached."""

_FACTORY_CACHING_INTERVAL = 12
"""int: Interval in hours defining the refresh rate of the cache."""

_KNOWN_ROUTES = sanitize_path(os.path.join(_HERE, 'knownroutes.json'))
"""str: Path to the file where routes are cached."""

_KNOWN_ROUTES_CACHING_INTERVAL = 12
"""int: Interval in hours defining the refresh rate of the cache."""

# API SETTINGS ----------------------------------------------------------------

_ROUTING_URL = 'http://router.project-osrm.org/route/v1/car/{0},{1};{2},{3}{4}'
_OVERPASS_URL = 'http://overpass-api.de/api/interpreter'
_LAST_REQUEST = 0
_USER_AGENT = ('FERTIGTEIL2.0 (prototype application within circular economy '
               'research project at TU Darmstadt, written in Python. Server is'
               'only used for testing purposes.) Please do not ban.')


# FUNCTION DEFINITIONS --------------------------------------------------------

def _query_distance_latlon(lat_a: float,
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


def _query_distance(loc_a: Tuple[float, float],
                    loc_b: Tuple[float, float],
                    url: str = _ROUTING_URL):
    """
    Queries the OSRM routing server for a distance between the two locations.
    Returns distance in kilometers.
    NOTE: Location tuples are defined as (lat, lon)!
    """
    lat_a, lon_a = loc_a
    lat_b, lon_b = loc_b
    return _query_distance_latlon(lat_a, lon_a, lat_b, lon_b, url=url)


def _get_concrete_landfills():
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
    print('[FT20 ROUTING] Querying for Concrete Landfill locations...')
    response = requests.get(_OVERPASS_URL,
                            params={'data': query},
                            headers=headers)
    data = response.json()
    return _sanitize_overpass_data(data)


def _get_concrete_factories():
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
    print('[FT20 ROUTING] Querying for Concrete Factory locations...')
    response = requests.get(_OVERPASS_URL,
                            params={'data': query},
                            headers=headers)
    data = response.json()
    return _sanitize_overpass_data(data)


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
        return _query_distance(loc_a, loc_b, url=url)
    else:
        time.sleep(1.0 - (time.time() - _LAST_REQUEST))
        return _query_distance(loc_a, loc_b, url=url)


def calculate_route(loc_a: Tuple[float, float],
                    loc_b: Tuple[float, float],
                    url: str = _ROUTING_URL,
                    interval: int = _KNOWN_ROUTES_CACHING_INTERVAL):
    """
    Check for a known route between the two locations and compute it if no
    known route exists.
    """
    # create file version to check against file
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    # get known routes and file version from file
    known_routes, version = _get_known_routes()
    loc_a_str = _loc_to_str(loc_a)
    loc_b_str = _loc_to_str(loc_b)
    route_a = loc_a_str + '->' + loc_b_str
    route_b = loc_b_str + '->' + loc_a_str
    if route_a not in known_routes:
        print('[FT20 ROUTING] Calculating car-route from '
              f'{loc_a} to {loc_b} using OSRM...')
        dist = get_distance(loc_a, loc_b)
        known_routes[route_a] = dist
        known_routes[route_b] = dist
        _update_known_routes(known_routes)
    else:
        last_refresh = _hours_between(now, version)
        if last_refresh > interval:
            print('[FT20 ROUTING] Calculating car-route from '
                  f'{loc_a} to {loc_b} using OSRM...')
            dist = get_distance(loc_a, loc_b)
            known_routes[route_a] = dist
            known_routes[route_b] = dist
            _update_known_routes(known_routes)
        else:
            dist = known_routes[route_a]
    return dist


def calculate_transporthistory(transporthistory: Sequence[Sequence]):
    """
    Calculate all missing distances in a transporthistory.
    """
    known_routes, version = _get_known_routes()
    for i, transport in enumerate(transporthistory):
        if int(transport[2]) == -1:
            loc_a = transport[0]
            loc_b = transport[1]
            dist = calculate_route(loc_a, loc_b)
            transporthistory[i][2] = dist
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
    distances = []
    # loop over repository components and create dict of unique locations
    for i, comp in enumerate(repository_components):
        loc_a = comp.location
        loc_b = site_location
        dist = calculate_route(loc_a, loc_b)
        distances.append(dist)
    return distances


def compute_landfill_distances(
        repository_components: List[RepositoryComponent],
        cache: str = _LANDFILL_CACHE,
        interval: int = _LANDFILL_CACHING_INTERVAL):
    # create file version to check against file
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    # read from file
    with open(cache, 'r', encoding='utf-8') as f:
        landfills = json.load(f)
    last_refresh = _hours_between(landfills['file_version'], now)
    print(('[FT20 ROUTING] Last refresh for Concrete Landfill cache was'
           f' {last_refresh} hours ago.'))
    if last_refresh > interval:
        # get all landfills across germany
        landfills = _get_concrete_landfills()
        landfills['file_version'] = now
        with open(cache, 'w', encoding='utf-8') as f:
            json.dump(landfills, f, ensure_ascii=False, indent=4)
    del landfills['file_version']

    # compute haversine distance for all locations to find nearest landfill
    landfill_distances = []
    for i, rc in enumerate(repository_components):
        loc_a = rc.transporthistory_parsed[0][0]
        lfdists = sorted([(k, _haversine(loc_a, landfills[k]))
                          for k in landfills.keys()], key=itemgetter(1))
        nlf = landfills[lfdists[0][0]]
        # compute car-routed distance to nearest landfill using OSRM
        dist = calculate_route(loc_a, nlf)
        landfill_distances.append(dist)

    return landfill_distances


def compute_factory_distance(
        loc: Tuple[float, float],
        cache: str = _FACTORY_CACHE,
        interval: int = _FACTORY_CACHING_INTERVAL):
    """
    Compute the distance to the nearest concrete factory based on a given
    location.
    """
    # create file version to check against file
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    # read from file
    with open(cache, 'r', encoding='utf-8') as f:
        factories = json.load(f)
    last_refresh = _hours_between(factories['file_version'], now)
    print(('[FT20 ROUTING] Last refresh for Concrete Factory cache was'
           f' {last_refresh} hours ago.'))
    if last_refresh > interval:
        # get all concrete factories across germany
        factories = _get_concrete_factories()
        factories['file_version'] = now
        with open(cache, 'w', encoding='utf-8') as f:
            json.dump(factories, f, ensure_ascii=False, indent=4)
    del factories['file_version']

    # compute haversine distance for all locations to find nearest landfill
    factory_distances = sorted([(k, _haversine(loc, factories[k]))
                                for k in factories.keys()], key=itemgetter(1))
    ncf = factories[factory_distances[0][0]]
    dist = calculate_route(loc, ncf)
    return dist


# UTILITIES -------------------------------------------------------------------

def _haversine(origin: Tuple[float, float], destination: Tuple[float, float]):
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
    return str(loc).strip('[]()')


def _str_to_loc(locstr: str):
    """
    Utility function to convert a string to a location-tuple with signature
    (lat, lon)
    """
    return [float(c.strip(' ')) for c in locstr[0].split(',')]


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


def _get_known_routes(known_routes_file: str = _KNOWN_ROUTES):
    """
    Utility function to retrieve known routes from json file.
    """
    # create file version to check against file
    version = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    # get known routes and file version from file
    with open(known_routes_file, 'r', encoding='utf-8') as f:
        known_routes = json.load(f)
    version = known_routes['file_version']
    del known_routes['file_version']
    return known_routes, version


def _update_known_routes(known_routes: dict,
                         known_routes_file: str = _KNOWN_ROUTES):
    version = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    known_routes['file_version'] = version
    with open(_KNOWN_ROUTES, 'w', encoding='utf-8') as f:
        json.dump(known_routes, f, ensure_ascii=False, indent=4)
    return True


def _hours_between(d1: str, d2: str):
    """
    Computes the hours between two file version strings for caching.
    """
    d1 = datetime.datetime.strptime(d1, '%Y-%m-%d-%H')
    d2 = datetime.datetime.strptime(d2, '%Y-%m-%d-%H')
    return ((d2 - d1).seconds / 3600)


# RUN SCRIPT ------------------------------------------------------------------

if __name__ == "__main__":
    # test osrm routing query
    loc_a = (49.863323, 8.678350)
    loc_b = (52.284726, 10.548717)
    print(f'Distance is {get_distance(loc_a, loc_b)} km.')

    # test overpass location query
    loc_a = (50.284726, 10.548717)
    landfills = _get_concrete_landfills()
    lfdists = sorted([(k, _haversine(loc_a, landfills[k]))
                      for k in landfills.keys()], key=itemgetter(1))
    nlf = lfdists[0]
    print(lfdists)
    print(nlf)
