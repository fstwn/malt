# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import json
from typing import Sequence, Tuple


# CLASS DEFINITION ------------------------------------------------------------

class DemandComponent(object):

    # FIELDS ------------------------------------------------------------------

    __objtype = ''
    __location = ()
    __date = ''
    __boundingbox = []
    __material = ''
    __geometry = ''

    # CONSTRUCTORS ------------------------------------------------------------

    def __init__(self,
                 objtype: str,
                 location: Tuple[float, float],
                 date: str,
                 boundingbox: Sequence[float],
                 material: str,
                 geometry: str):

        self.objtype = objtype
        self.location = location
        self.date = date
        self.boundingbox = boundingbox
        self.material = material
        self.geometry = geometry

    @classmethod
    def CreateFromDict(cls, dataset):
        return cls(
            dataset['type'],
            dataset['location'],
            dataset['date'],
            dataset['boundingbox'],
            dataset['material'],
            dataset['geometry']
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

    # LOCATION

    @property
    def location(self):
        return self.__location

    @location.setter
    def location(self, location: Sequence[float]):
        self.__location = location

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

    # MATERIAL

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, material: str):
        self.__material = material

    # GEOMETRY

    @property
    def geometry(self):
        return self.__geometry

    @geometry.setter
    def geometry(self, geometry: str):
        self.__geometry = geometry

    # DATA

    @property
    def DataDict(self):
        return {
            'type': self.objtype,
            'location': self.location,
            'date': self.date,
            'boundingbox': self.boundingbox,
            'material': self.material,
            'geometry': self.geometry
        }

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
