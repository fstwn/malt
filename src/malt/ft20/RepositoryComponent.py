# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import itertools
import json
import uuid
from typing import Sequence, Tuple


# LOCAL MODULE IMPORTS --------------------------------------------------------

from malt.hopsutilities import validate_uuid


# CLASS DEFINITION ------------------------------------------------------------

class RepositoryComponent(object):

    # FIELDS ------------------------------------------------------------------

    __objtype = ''
    __availability = False
    __location = ()
    __dummy = True
    __date = ''
    __boundingbox = []
    __pointcloud = ''
    __ifcobject = ''
    __dom = -1
    __material = ''
    __process = []
    __transporthistory = []
    __componenthistory = []
    __geometry = ''
    __uid = ''

    # CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 objtype: str,
                 availability: bool,
                 location: Tuple[float, float],
                 dummy: bool,
                 date: str,
                 boundingbox: Sequence[float],
                 pointcloud: str,
                 ifcobject: str,
                 dom: int,
                 material: str,
                 process: Sequence[Sequence[str]],
                 transporthistory: Sequence[Sequence[str]],
                 componenthistory: Sequence[str],
                 geometry: str,
                 uid: str = ''):

        self.objtype = objtype
        self.availability = availability
        self.location = location
        self.dummy = dummy
        self.date = date
        self.boundingbox = boundingbox
        self.pointcloud = pointcloud
        self.ifcobject = ifcobject
        self.dom = dom
        self.material = material
        self.process = process
        self.transporthistory = self._validate_transporthistory(
            transporthistory)
        self.componenthistory = componenthistory
        self.geometry = geometry

        if uid:
            if not validate_uuid(uid):
                raise ValueError(f'Supplied uid {uid} is not a valid UUID!')
            else:
                self.__uid = uid
        else:
            self.__uid = str(uuid.uuid4())

    @classmethod
    def CreateFromDict(cls, dataset):
        return cls(
            dataset['type'],
            dataset['availability'],
            dataset['location'],
            dataset['dummy'],
            dataset['date'],
            dataset['boundingbox'],
            dataset['pointcloud'],
            dataset['ifcobject'],
            dataset['dom'],
            dataset['material'],
            dataset['process'],
            dataset['transporthistory'],
            dataset['componenthistory'],
            dataset['geometry'],
            uid=dataset['uid'] if 'uid' in dataset.keys() else ''
        )

    @classmethod
    def CreateFromJSON(cls, jsonstr):
        return cls.CreateFromDict(json.loads(jsonstr))

    # PROPERTIES- -------------------------------------------------------------

    # OBJTYPE

    @property
    def objtype(self):
        return self.__objtype

    @objtype.setter
    def objtype(self, objtype: str):
        self.__objtype = objtype

    # AVAILABILITY

    @property
    def availability(self):
        return self.__availability

    @availability.setter
    def availability(self, availability: bool):
        self.__availability = availability

    # LOCATION

    @property
    def location(self):
        return self.__location

    @location.setter
    def location(self, location: Sequence[float]):
        if len(location) != 2:
            raise
        self.__location = tuple(location)

    # DUMMY

    @property
    def dummy(self):
        return self.__dummy

    @dummy.setter
    def dummy(self, dummy: bool):
        self.__dummy = dummy

    # DATE

    @property
    def date(self):
        return self.__date

    @date.setter
    def date(self, date: str):
        self.__date = date

    # BOUNDINGBOX

    @property
    def boundingbox(self):
        return self.__boundingbox

    @boundingbox.setter
    def boundingbox(self, boundingbox: Sequence[float]):
        if len(boundingbox) != 3:
            raise
        self.__boundingbox = boundingbox

    # POINTCLOUD

    @property
    def pointcloud(self):
        return self.__pointcloud

    @pointcloud.setter
    def pointcloud(self, pointcloud: str):
        self.__pointcloud = pointcloud

    # IFCOBJECT

    @property
    def ifcobject(self):
        return self.__ifcobject

    @ifcobject.setter
    def ifcobject(self, ifcobject: str):
        self.__ifcobject = ifcobject

    # DOM

    @property
    def dom(self):
        return self.__dom

    @dom.setter
    def dom(self, dom: int):
        self.__dom = dom

    # MATERIAL

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, material: str):
        self.__material = material

    # PROCESS

    @property
    def process(self):
        return self.__process

    @process.setter
    def process(self, process):
        self.__process = process

    # TRANSPORTHISTORY

    @property
    def transporthistory(self):
        return self.__transporthistory

    @transporthistory.setter
    def transporthistory(self, transporthistory):
        self.__transporthistory = transporthistory

    @property
    def transporthistory_parsed(self):
        return self._parse_transporthistory(self.__transporthistory)

    # COMPONENTHISTORY

    @property
    def componenthistory(self):
        return self.__componenthistory

    @componenthistory.setter
    def componenthistory(self, componenthistory):
        self.__componenthistory = componenthistory

    # GEOMETRY

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self, geometry: str):
        self.__geometry = geometry

    # UUID

    @property
    def uid(self):
        return self.__uid

    # DATA

    @property
    def DataDict(self):
        datadict = {}
        datadict['type'] = self.objtype
        datadict['availability'] = self.availability
        datadict['location'] = self.location
        datadict['dummy'] = self.dummy
        datadict['date'] = self.date
        datadict['boundingbox'] = self.boundingbox
        datadict['pointcloud'] = self.pointcloud
        datadict['ifcobject'] = self.ifcobject
        datadict['dom'] = self.dom
        datadict['material'] = self.material
        datadict['process'] = self.process
        datadict['transporthistory'] = self.transporthistory
        datadict['componenthistory'] = self.componenthistory
        datadict['geometry'] = self.geometry
        datadict['uid'] = self.uid
        return datadict

    @property
    def JSON(self):
        return json.dumps(self.DataDict)

    # REPRESENTATION ----------------------------------------------------------

    def __str__(self):
        return self.JSON

    def __repr__(self):
        return self.JSON

    def ToString(self):
        return self.JSON

    # UTILS -------------------------------------------------------------------

    def _validate_transporthistory(self, transporthistory: Sequence[Sequence]):
        """
        Validate a transporthistory by ensuring all distances are present
        and the signature matches that of the API server.
        """
        from malt.ft20 import calculate_transporthistory
        if (all(isinstance(obj, str)
                for obj in itertools.chain.from_iterable(transporthistory))):
            transporthistory = self._parse_transporthistory(transporthistory)
        transporthistory = calculate_transporthistory(transporthistory)
        transporthistory = self._str_transporthistory(transporthistory)
        return transporthistory

    def _parse_transporthistory(self,
                                transporthistory: Sequence[Sequence[str]]):
        """
        Parse a transporthistory from all-string signature into
        [[[float, float], [float, float], int, str]] signature.
        """
        new_history = []
        for obj in transporthistory:
            loc_a = [float(c.strip(' ')) for c in obj[0].split(',')]
            loc_b = [float(c.strip(' ')) for c in obj[1].split(',')]
            dist = float(obj[2])
            ttype = obj[3]
            new_history.append([loc_a, loc_b, dist, ttype])
        return new_history

    def _str_transporthistory(self, transporthistory: Sequence[Sequence]):
        """
        Convert a transporthistory to all string signature
        [[str, str, str, str]]
        """
        new_history = []
        for obj in transporthistory:
            loc_a = str(obj[0]).strip('[]')
            loc_b = str(obj[1]).strip('[]')
            dist = str(obj[2])
            ttype = str(obj[3])
            new_history.append([loc_a, loc_b, dist, ttype])
        return new_history
