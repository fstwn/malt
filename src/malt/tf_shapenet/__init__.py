# Classification and clustering of 2d-outlines (shapes) using tensorflow.
# Script based on the materials of the MIT Workshop "Kintsugi, Upcycling and
# Machine Learning", held in summer 2020.
#
# Workshop Materials & Examples by Daniel Marshall & Yijiang Huang
#
# For references see:
# https://architecture.mit.edu/subject/summer-2020-4181
# https://docs.google.com/document/d/1qO5-4QdBO_dp3kl_R9rK6IvAdyQ0pFmJr8JDxQLjWc4/edit?usp=sharing
# https://architecture.mit.edu/sites/architecture.mit.edu/files/attachments/course/20SU-4.181_syll_marshall%2Bmueller.pdf
#
# Adapted by Max Eschenbach, DDU, TU-Darmstadt (2021)

from .shapenet import (initial_train, # NOQA401
                       load_and_train,
                       forward_pass)
