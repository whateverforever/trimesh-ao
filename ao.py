#!/usr/bin/env python3

import numpy as np
import trimesh
import trimesh.sample as sample
from scipy.spatial import KDTree

model = trimesh.load("untitled.ply")
assert isinstance(model, trimesh.Trimesh)
model.fix_normals()

NDIRS = 32
sphere_pts = sample.sample_surface_sphere(NDIRS)
stree = KDTree(sphere_pts)

# at each vertex, we're sending out this many rays
nrays = int(NDIRS // 2)
_, closest_dirs = stree.query(model.vertex_normals, nrays)
# print("verts", len(model.vertices), "dirs", len(closest_dirs))

origins = np.array([vert for vert in model.vertices for _ in range(nrays)])
directions = [sphere_pts[dir] for dirs in closest_dirs for dir in dirs]
assert len(origins) == len(directions)
# print("origins", origins[:100])
# print("directions", directions[:100])

locations, index_ray, index_tri = model.ray.intersects_location(
    ray_origins=origins, ray_directions=directions
)

# don't check infinitely long rays
succ_origs = origins[index_ray]
succ_locs = locations
distances = np.linalg.norm(succ_origs - succ_locs, axis=1)
index_ray = index_ray[distances < 0.1 * model.scale]

# count up how many of the rays per location were stopped
# by something
counts = np.zeros(len(origins))
counts[index_ray] += 1
counts = counts.reshape(-1, nrays)

counts_verts = np.sum(counts, axis=1)
counts_verts /= np.max(counts_verts)
counts_verts *= 255
counts_verts = 255 - counts_verts.astype(int).reshape(-1,1)
# print("counts_verts", counts_verts.shape)

colors = np.hstack([counts_verts, counts_verts, counts_verts, [[255]] * len(counts_verts)])
# print("colors", colors)

model.visual.vertex_colors = colors
model.show()
