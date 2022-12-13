# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import json
import uuid
from typing import List, Sequence


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
                 objtype,
                 availability,
                 location,
                 dummy,
                 date,
                 boundingbox,
                 pointcloud,
                 ifcobject,
                 dom,
                 material,
                 process,
                 transporthistory,
                 componenthistory,
                 geometry,
                 uid=None):

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
        self.transporthistory = transporthistory
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
            uid=dataset['uid']
        )

    @classmethod
    def CreateFromJSON(cls, jsonobj):
        return cls.CreateFromDict(json.loads(jsonobj))

    # PROPERTIES --------------------------------------------------------------

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
        self.__location = location

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
    def Data(self):
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
        return json.dumps(self.Data)

    # REPRESENTATION ----------------------------------------------------------

    def __str__(self):
        return f'<RepositoryComponent [{self.uid}]>'

    def __repr__(self):
        return self.__str__()

    def ToString(self):
        return self.__str__()

    # UTILS -------------------------------------------------------------------

    def _parse_transporthistory(self, transporthistory: List[List[str]]):
        new_history = []
        for obj in transporthistory:
            loc_a = tuple([float(c.strip(' ')) for c in obj[0].split(',')])
            loc_b = tuple([float(c.strip(' ')) for c in obj[1].split(',')])
            dist = float(obj[2])
            ttype = obj[3]
            new_history.append((loc_a, loc_b, dist, ttype))
        return new_history

    def _str_transporthistory(self, transporthistory):
        new_history = []
        for obj in transporthistory:
            loc_a = str(obj[0]).strip('()')
            loc_b = str(obj[1]).strip('()')
            dist = str(obj[2])
            ttype = str(obj[3])
            new_history.append([loc_a, loc_b, dist, ttype])
        return new_history
