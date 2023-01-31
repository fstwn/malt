# Implementation of the paper:
#
# Cohen-Steiner, D., Alliez, P., Desbrun, M., 2004.
# Variational shape approximation.
# ACM Trans. Graph. 23, 905–914.
# https://doi.org/10.1145/1015706.1015817
#
# Original implementation by GitHub User romain22222
# https://github.com/romain22222/PROJ602-Variational-shape-approximation
#
# Re-implemented for use with Rhino.Inside.CPython
# by Max Eschenbach, DDU, TU-Darmstadt

# PYTHON STANDARD LIBRARY IMPORTS ---------------------------------------------

import itertools
import random
from functools import total_ordering
from queue import PriorityQueue
from typing import Dict, Set, Sequence, Tuple


# THIRD PARTY LIBRARY IMPORTS -------------------------------------------------

import numpy as np


# CLASS DEFINITIONS -----------------------------------------------------------

class Region(object):
    """
    Class for storing region information
    """

    index = None
    faceids = None
    normal = None

    def __init__(self, index: str, faceids: Sequence[int], normal=None):
        self.index = index
        self.faceids = faceids
        self.normal = normal

    def __str__(self):
        return (self.index + " : " + str(self.faceids) +
                " - " + str(self.normal))

    def __len__(self):
        return len(self.faceids)


@total_ordering
class QueueElement(object):
    def __init__(self, error: float, region_index: str, index: int):
        self.error = error
        self.region_index = region_index
        self.index = index

    def __str__(self):
        return ("Face " + str(self.index) + " of region " +
                str(self.region_index) + " : " + str(self.error))

    def __gt__(self, other):
        return self.error > other.error

    def __eq__(self, other):
        return self.error == other.error


class VariationalSurfaceApproximation(object):
    """
    Solver class
    """

    _iterations = 1

    _vertices = None
    _faces = None
    _normals = None
    _edges = None
    _areas = None
    _adjacency = None

    _nbproxies = None
    _regions = None

    _last_region_id = 0

    def _generate_id(self):
        self._last_region_id += 1
        return str(self._last_region_id)

    def __init__(self, V: np.array, F: np.array, numregions: int = 2):
        # first we have to set vertices and faces
        self._vertices = V
        self._faces = F
        # then we have to compute a bunch of stuff
        self._areas = self.compute_face_areas()
        self._normals = self.compute_face_normals()
        self._edges = self.compute_face_edges()
        self._adjacency = self.compute_adjacency()
        # last but not least we generate a bunch of randomized regions
        self._regions = self.generate_regions(numregions)

    @staticmethod
    def compute_triangular_face_area(vect1, vect2):
        return np.linalg.norm(
            np.array(
                [
                    vect1[1] * vect2[2] - vect1[2] * vect2[1],
                    vect1[2] * vect2[0] - vect1[0] * vect2[2],
                    vect1[0] * vect2[1] - vect1[1] * vect2[0]
                ]
            )
        ) * 0.5

    @staticmethod
    def orders(vertex_a, vertex_b) -> bool:
        if vertex_a[0] != vertex_b[0]:
            return vertex_a[0] > vertex_b[0]
        if vertex_a[1] != vertex_b[1]:
            return vertex_a[1] > vertex_b[1]
        return vertex_a[2] > vertex_b[2]

    @staticmethod
    def costr(num) -> str:
        return str(num) if num < 0 else "+" + str(num)

    @staticmethod
    def coords_to_str(c1, c2) -> str:
        vsa = VariationalSurfaceApproximation
        if vsa.orders(c1, c2):
            tmp = c1
            c1 = c2
            c2 = tmp
        return str(vsa.costr(c1[0]) + vsa.costr(c2[0]) + vsa.costr(c1[1]) +
                   vsa.costr(c2[1]) + vsa.costr(c1[2]) + vsa.costr(c2[2]))

    def compute_face_areas(self):
        areas = []
        for face in self._faces:
            areas.append(
                self.compute_triangular_face_area(
                    self._vertices[face[1]] - self._vertices[face[0]],
                    self._vertices[face[2]] - self._vertices[face[0]]
                )
            )
        return areas

    def compute_face_normals(self):
        normals = []
        for face in self._faces:
            U = self._vertices[face[1]] - self._vertices[face[0]]
            V = self._vertices[face[2]] - self._vertices[face[0]]
            normals.append(
                (
                    U[1] * V[2] - U[2] * V[1],
                    U[2] * V[0] - U[0] * V[2],
                    U[0] * V[1] - U[1] * V[0]
                )
            )
        return normals

    def compute_face_edges(self):
        edges = []
        correspondence = {}
        for face in self._faces:
            fE = []
            for i in range(3):
                key = self.coords_to_str(self._vertices[face[i]],
                                         self._vertices[face[i - 1]])
                idE = correspondence.get(key, None)
                if idE is None:
                    correspondence[key] = len(correspondence.keys())
                    idE = correspondence[key]
                fE.append(idE)
            edges.append(fE)
        return edges

    def compute_adjacency(self):
        if self._edges is None:
            self._edges = self.compute_face_edges()
        adjacency = [set() for i in range(len(self._faces))]
        corrEdges = {}
        for i in range(len(self._edges)):
            for j in self._edges[i]:
                if corrEdges.get(j, None) is not None:
                    adjacency[corrEdges[j]].add(i)
                    adjacency[i].add(corrEdges[j])
                else:
                    corrEdges[j] = i
        return adjacency

    def generate_regions(self, n: int) -> Dict[str, Region]:
        """
        Create a number nb of regions with a random face
        """
        faces = self._faces[:]
        regions = {}
        faceDrawn = []
        for i in range(n):
            face = random.randrange(len(faces))
            while face in faceDrawn:
                face = random.randrange(len(faces))
            rid = self._generate_id()
            regions[rid] = Region(rid, [face])
            faceDrawn.append(face)
        return regions

    def compute_seed_faces(self):
        """
        Recalculate the new regions, and assign them a seed face
        """
        regions = {}
        for region in self._regions.values():

            tmp = self.calculate_new_elements_of_queue(
                PriorityQueue(),
                region,
                region.faceids
            )

            errors = []
            while not tmp.empty():
                errors.append(tmp.get())

            errors.sort(key=lambda x: x.error)
            seedFaceIndex = errors.pop(0).index
            region = Region(region.index,
                            [seedFaceIndex],
                            region.normal)
            regions[region.index] = region

        self._regions = regions

    def insert_region(self, region: Region):
        """
        Insert a new region into the list of regions
        """
        self._regions[region.index] = region

    def insert_regions(self, regions: Dict[str, Region]):
        """
        Insert multiple new regions into the list of regions
        """
        self._regions.update(regions)

    def update_region_normals(self):
        """
        Updates the normals of all current regions.
        """
        for region in self._regions.values():
            region.normal = self.compute_region_normal(region.faceids)

    def compute_region_normal(self, faceids: Sequence[int]) -> np.array:
        """
        Calculates and returns the normal of the region.
        """
        normal = np.array([0, 0, 0])
        for index in faceids:
            normal = np.add(normal, self._normals[index])
        if normal[0] != 0 or normal[1] != 0 or normal[2] != 0:
            normal = normal / np.linalg.norm(normal)
        return normal

    def calculate_new_elements_of_queue(self,
                                        queue: PriorityQueue,
                                        region: Region,
                                        faceids: Sequence[int],
                                        combine: bool = False,
                                        merge_params: dict = {}):
        """
        Adds the given faces to the queue

        Special behavior if combine is True: Finds the two regions to merge
        """
        region_index = region.index
        for fid in faceids:
            area = self._areas[fid]
            normal = self._normals[fid]
            normalError = normal - region.normal

            error = abs(((np.linalg.norm(normalError)) ** 2) * area)

            if combine:
                if error > merge_params["max_error"] and merge_params["i"] > 0:
                    break
                else:
                    merge_params["regions_to_combine"] = merge_params["merged_region"] # NOQA501
                    merge_params["max_error"] = error
            else:
                queue.put(QueueElement(error, region_index, fid))

        return merge_params if combine else queue

    def update_queue(self,
                     queue: PriorityQueue,
                     region: Region,
                     faceids: Sequence[int]) -> PriorityQueue:
        return self.calculate_new_elements_of_queue(
            queue,
            region,
            faceids
        )

    def build_queue(self, regions: Dict[str, Region]):
        """
        Builds a queue based on the input regions, containing all seed faces'
        adjacent faces.
        """
        queue = PriorityQueue()
        assigned_ids = set()
        # loop over all established regions
        for region in regions.values():
            # get seed face index, which is always the first one since the
            # regions should only have one face assigned at this point in time
            seed_index = region.faceids[0]
            # add seed index to the set of assigned face ids
            assigned_ids.add(seed_index)
            # update the queue using the adjacency information
            self.update_queue(
                queue,
                region,
                self._adjacency[seed_index]
            )
        return queue, assigned_ids

    @staticmethod
    def find_worst(queue_list: Sequence[QueueElement]):
        """
        Searches the queue for the maximum error element, removes it from the
        list and returns it
        """
        index = 0
        maxE = -np.inf
        for i in range(len(queue_list)):
            if queue_list[i].error > maxE:
                maxE = queue_list[i].error
                index = i
        return queue_list.pop(index)

    def assign_faces_to_regions(self,
                                queue: PriorityQueue,
                                assigned_ids: Set[int]):
        """
        Distributes the faces to the different regions according to their
        proximity to them
        """
        queue_list = []

        while not queue.empty():
            prio_elem = queue.get()
            faceid = prio_elem.index
            if faceid not in assigned_ids:
                queue_list.append(prio_elem)
                region = self._regions.get(prio_elem.region_index)
                region.faceids.append(faceid)
                assigned_ids.add(faceid)
                new_adj_faces = set(self._adjacency[faceid])
                new_adj_faces -= assigned_ids

                self.update_queue(
                    queue,
                    region,
                    new_adj_faces
                )
        try:
            worst = self.find_worst(queue_list)
        except IndexError:
            # If all elements are assigned, no queue_list is filled
            # (case where regions = number of faces)
            random_reg = random.randrange(
                0, len(self._regions) - 1)

            random_reg_face = random.randrange(
                0, len(self._regions[random_reg]) - 1)

            worst = QueueElement(
                0.0,
                self._regions[random_reg].index,
                self._regions[random_reg].faceids[random_reg_face]
            )

        return worst

    def find_region(self, region_index: str):
        """
        Find a region from the list of regions by index.
        """
        return self._regions.get(region_index)

    def remove_region(self, region_index: str):
        """
        Remove a region from the list of regions
        """
        del self._regions[region_index]

    def assign_faces_to_split_regions(self,
                                      queue: PriorityQueue,
                                      split_regions: Dict[str, Region],
                                      assigned_ids: Set[int],
                                      old_faces: Sequence[int]):
        """
        Distributes the faces of the worst region to two other regions
        """

        region_domain = frozenset(old_faces)

        while not queue.empty():
            prio_elem = queue.get()
            if prio_elem.index not in region_domain:
                continue
            faceid = prio_elem.index
            if faceid not in assigned_ids:
                region_index = prio_elem.region_index
                for region in split_regions.values():
                    if region_index == region.index:
                        region.faceids.append(faceid)
                        assigned_ids.add(faceid)
                        adjacency_set = set(self._adjacency[faceid])
                        adjacency_set &= region_domain
                        adjacency_set -= assigned_ids
                        if adjacency_set:
                            self.update_queue(
                                queue,
                                region,
                                adjacency_set
                            )

        return split_regions

    def split_region(self, worst: QueueElement):
        # retrieve worst region
        worst_region = self.find_region(worst.region_index)

        # create ids for the newly split regions
        reg_idx_a = self._generate_id()
        reg_idx_b = self._generate_id()

        # get old faces
        old_faces = worst_region.faceids

        # set seed faces for newly split regions
        seed_a = old_faces[0]
        seed_b = worst.index

        # 1. build new regions
        # 2. compute normals for the new regions
        # 3. build new queue based on new regions

        # build new regions
        split_regions = {}
        reg_a = Region(
            reg_idx_a,
            [seed_a],
            self.compute_region_normal([seed_a])
        )
        reg_b = Region(
            reg_idx_b,
            [seed_b],
            self.compute_region_normal([seed_b])
        )
        split_regions[reg_idx_a] = reg_a
        split_regions[reg_idx_b] = reg_b

        # build queue based on these new regions
        queue, assigned_ids = self.build_queue(split_regions)

        # assign faces to worst region
        split_regions = self.assign_faces_to_split_regions(queue,
                                                           split_regions,
                                                           assigned_ids,
                                                           old_faces)
        # remove worst region
        self.remove_region(worst_region.index)

        # add new regions
        self.insert_regions(split_regions)

    def find_adjacent_regions(self, regions: Dict[str, Region]):
        adjacent_regions = []
        all_region_edges = []
        for region in regions.values():
            region_index = region.index
            region_edges = []
            for i in region.faceids:
                region_edges.extend(self._edges[i])
            all_region_edges.append([region_index, set(region_edges)])
        for region_a, region_b in itertools.combinations(all_region_edges, 2):
            if region_a[1].intersection(region_b[1]):
                adjacent_regions.append([region_a[0], region_b[0]])

        return adjacent_regions

    def find_regions_to_combine(self,
                                regions: Dict[str, Region],
                                adj_regions: [Sequence[Sequence[Region]]]):
        
        # init merge params
        merge_params = {
            "max_error": -np.inf,
            "regions_to_combine": None,
            "merged_region": None,
            "i": None
        }

        regions_to_del = adj_regions[0]

        for i, adjacent in enumerate(adj_regions):
            region_a = regions.get(adjacent[0])
            region_b = regions.get(adjacent[1])

            # create new regions dict and add merged region
            reg_id = self._generate_id()
            reg_faces = region_a.faceids + region_b.faceids
            reg_normal = self.compute_region_normal(reg_faces)
            merged_region = Region(reg_id, reg_faces, reg_normal)

            merge_params = {
                "max_error": merge_params["max_error"],
                "regions_to_combine": merge_params["regions_to_combine"],
                "merged_region": merged_region,
                "i": i
            }

            merge_params = self.calculate_new_elements_of_queue(
                PriorityQueue(),
                merged_region,
                merged_region.faceids,
                combine=True,
                merge_params=merge_params
            )

            if merge_params["regions_to_combine"] == merged_region:
                regions_to_del = adjacent

        # remove the regions to delete
        [self.remove_region(idx) for idx in regions_to_del]

        # insert merged regions
        self.insert_region(merge_params["regions_to_combine"])

    def extract_region_mesh(self, region: Region) -> Tuple[np.array, np.array]:

        region_face_vertices = [self._faces[i] for i in region.faceids]

        region_vertices = set([i for sublist in region_face_vertices
                               for i in sublist])

        vertex_map = dict(list(zip(region_vertices,
                                   list(range(len(region_vertices))))))

        new_face_ids = []
        for item in region_face_vertices:
            new_face_ids.append([vertex_map[i] for i in item])
        new_vertices = {}
        for k, v in vertex_map.items():
            new_vertices[v] = self._vertices[k]
        new_vertices = list(new_vertices.values())

        return np.array(new_vertices), np.array(new_face_ids)

    def Solve(self, n: int = 1):
        """
        Solve for n iterations.
        """
        for i in range(n):
            # Update regions (add normals)
            self.update_region_normals()

            # We collect the seeds from each region
            self.compute_seed_faces()

            # We build a queue containing all the faces adjacent to the seeds,
            # and the list of seeds
            queue, assigned_ids = self.build_queue(self._regions)

            # We will then distribute the faces according to the region that
            # gives them the smallest error
            worst = self.assign_faces_to_regions(queue, assigned_ids)

            # We get the worst side of the worst region, and with this one,
            # we will split the worst region in two
            self.split_region(worst)

            # NOTE: we will have one region more now!

            # After that we recover the adjacent regions
            adjacent_regions = self.find_adjacent_regions(self._regions)

            # And merge the two best adjacent regions
            self.find_regions_to_combine(self._regions, adjacent_regions)

            # NOTE: we should now have the same number of regions as initially

        region_meshes = []
        for region in self._regions.values():
            V, F = self.extract_region_mesh(region)
            region_meshes.append((V, F))

        return region_meshes
