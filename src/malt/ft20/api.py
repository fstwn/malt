# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import os


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import requests
import yaml


# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt.hopsutilities import sanitize_path, slash_join
from malt.ft20.RepositoryComponent import RepositoryComponent


# API CONFIG ------------------------------------------------------------------

_HERE = os.path.dirname(sanitize_path(__file__))
_CONFIG_FILE = sanitize_path(os.path.join(_HERE, 'config.yml'))

_BASE_URL = ''
_HOSTNAME = ''
_REALM_NAME = ''
_CLIENT_ID = ''
_CLIENT_SECRET = ''
_TOKEN_URL = ''
_ACCESS_TOKEN = ''

with open(_CONFIG_FILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    _BASE_URL = config['base_url']
    _HOSTNAME = config['hostname']
    _REALM_NAME = config['realm_name']
    _CLIENT_ID = config['client_id']
    _CLIENT_SECRET = config['client_secret']
    _TOKEN_URL = config['token_url']


# FUNCTION DEFINITIONS --------------------------------------------------------

def _get_headers(token=_ACCESS_TOKEN):
    """
    Obtain correct headers for performing a request to FARO Component
    Repository API.
    """
    # if no token exists, we need to authenticate first
    if not token:
        token = _get_access_token()
    # set header with authentication token
    return {'Authorization': f'Bearer {token}'}


def _get_access_token(url=_TOKEN_URL,
                      client_id=_CLIENT_ID,
                      client_secret=_CLIENT_SECRET):
    """
    Obtain an Access Token for the FARO Component Repository API.
    """
    # make authentication request
    print('[FARO-API] Obtaining authentication token...')
    response = requests.post(
        url,
        data={'grant_type': 'client_credentials'},
        auth=(client_id, client_secret)
    )

    # store token during runtime
    token = response.json()['access_token']

    global _ACCESS_TOKEN
    _ACCESS_TOKEN = token

    return token


def _make_request(method, endpoint, params=None, data=None):
    """
    Perform a request to the FARO Component Repository API
    """
    # create correct url
    global _BASE_URL
    url = slash_join(_BASE_URL, endpoint)

    # set header with authentication token
    global _ACCESS_TOKEN
    headers = _get_headers(token=_ACCESS_TOKEN)

    # perform request
    print(f'[FARO-API] Performing {method} request using token...')
    response = requests.request(method=method,
                                url=url,
                                params=params,
                                data=data,
                                headers=headers)

    # if response has status code for Unauthorized request, get new token
    # and try again one time
    if response.status_code == 401:
        print('[FARO-API] Request failed, retrying...')
        headers = _get_headers(token='')
        response = requests.request(method=method,
                                    url=url,
                                    params=params,
                                    data=data,
                                    headers=headers)

    return response


def get_all_objects():
    """
    Get all objects from the FARO Component Repository.
    """
    endpoint = 'objects/all'
    response = _make_request('GET', endpoint)
    resp_json = response.json()
    print(f'[FARO-API] Found {len(resp_json)} components on server.')
    return [RepositoryComponent.CreateFromDict(obj)
            for obj in resp_json]


def get_object(uid: str):
    """
    Get an object from the FARO Component Repository by its UID.
    """
    endpoint = f'objects/get/{uid}'
    response = _make_request('GET', endpoint)
    if response.status_code != 200:
        print(f'[FARO-API] Could not retrieve component with uid {uid}!')
        return (False, None)
    return (True, RepositoryComponent.CreateFromDict(response.json()))


def create_object(component: RepositoryComponent):
    """
    Create an object on the FARO Component Repository.
    """
    # test if component already exists on server
    if get_object(component.uid)[0]:
        raise ValueError(f'Component with UID {component.uid} already exists!')

    # create object
    endpoint = 'objects/create_with_uid'
    payload = component.JSON
    response = _make_request('POST', endpoint, data=payload)
    if response.status_code == 200:
        print(f'[FARO-API] Successfully created component {component.uid}.')
    return response.json()


def update_object(component: RepositoryComponent):
    """
    Update an object on the FARO Component Repository.
    """
    # test if component already exists on server
    if not get_object(component.uid)[0]:
        raise ValueError(f'Component with UID {component.uid} does not exist!')

    # create object
    endpoint = f'objects/update/{component.uid}'
    payload = component.JSON
    response = _make_request('POST', endpoint, data=payload)

    if response.status_code == 200:
        print(f'[FARO-API] Successfully updated component {component.uid}.')

    return response.json()


def delete_object(uid: str):
    """
    Delete an object from the FARO Component Repository by its UID.
    """
    endpoint = f'objects/delete/{uid}'
    response = _make_request('DELETE', endpoint)
    if response.status_code != 200:
        return False
    print(f'[FARO-API] Successfully deleted component {uid}.')

    return True


# RUN SCRIPT ------------------------------------------------------------------

if __name__ == "__main__":
    testdata = {
        "type": "Stuetze",
        "availability": True,
        "location": [52.284726, 10.548717],
        "dummy": True,
        "date": "20.07.2023",
        "boundingbox": [0.4, 0.4, 3.2],
        "pointcloud": "",
        "ifcobject": "",
        "dom": 2004,
        "material": "1.4.01 (Beton 30/37)",
        "process": [],
        "transporthistory": [
            ['52.284726, 10.548717',
             '49.863323, 8.678350',
             '-1',
             'Schwertransport']
        ],
        "componenthistory": [],
        "geometry": "",
        "uid": "82cedaaf-44cc-4956-88c6-865cd871a814",
    }
    testcomp = RepositoryComponent.CreateFromDict(testdata)
    create_object(testcomp)
    print(get_object(
        '82cedaaf-44cc-4956-88c6-865cd871a814')[1].transporthistory)
    delete_object("82cedaaf-44cc-4956-88c6-865cd871a814")
