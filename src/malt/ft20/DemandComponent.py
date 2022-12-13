# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import json


# CLASS DEFINITION ------------------------------------------------------------

class DemandComponent(object):

    objtype = ''
    location = None
    date = ''
    boundingbox = None
    material = ''
    geometry = ''

    def __init__(self,
                 objtype,
                 location,
                 date,
                 boundingbox,
                 material,
                 geometry):

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
    def CreateFromJSON(cls, jsonobj):
        return cls.CreateFromDict(json.loads(jsonobj))

    @property
    def Data(self):
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
        return json.dumps(self.Data)

    def __str__(self):
        return f'<DemandComponent [{self.objtype}]>'

    def __repr__(self):
        return self.__str__()

    def ToString(self):
        return self.__str__()
