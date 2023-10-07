#!/usr/bin/env python3

import numpy as np
import trimesh
import trimesh.sample as sample

model = trimesh.load("suzanne1.ply", force="mesh")
# model = trimesh.load("untitled.ply", force="mesh")
assert isinstance(model, trimesh.Trimesh)
# model.fix_normals()

# how many rays to send out from each vertex
NDIRS = 64
# how far away do surfaces still block the light?
# relative to the model diagonal
RELSIZE = 0.05

# TODO: replace with hinter sampling
sphere_pts, _ = sample.sample_surface_even(trimesh.primitives.Sphere(), count=NDIRS)


# TODO: benchmark against KDTree, seems slower and more memory
# We select which rays to send by checking which directions
# are in the same hemisphere as the respective model normal
normal_dir_similarities = model.vertex_normals @ sphere_pts.T
assert normal_dir_similarities.shape[0] == len(model.vertex_normals)
assert normal_dir_similarities.shape[1] == len(sphere_pts)

normal_dir_similarities[normal_dir_similarities <= 0] = 0
normal_dir_similarities[normal_dir_similarities > 0] = 1

# for each vertex, we get multiple directions
vert_idxs, dir_idxs = np.where(normal_dir_similarities)
del normal_dir_similarities

normals = model.vertex_normals[vert_idxs]
origins = model.vertices[vert_idxs] + normals * model.scale * 0.0001
directions = sphere_pts[dir_idxs]
assert len(origins) == len(directions)
#print("origins", origins[:100])
#print("directions", directions[:100])

hit_pts, idxs_rays, _ = model.ray.intersects_location(
    ray_origins=origins, ray_directions=directions
)

# don't check infinitely long rays
succ_origs = origins[idxs_rays]
distances = np.linalg.norm(succ_origs - hit_pts, axis=1)
# print("num rays before filter", len(idxs_rays))
idxs_rays = idxs_rays[distances < RELSIZE * model.scale]
# print("num rays after  filter", len(idxs_rays))

idxs_orig = vert_idxs[idxs_rays]
uidxs, uidxscounts = np.unique(idxs_orig, return_counts=True)
assert len(uidxs) == len(uidxscounts)

counts_verts = np.zeros(len(model.vertices))
counts_verts[uidxs] = uidxscounts
counts_verts /= np.max(counts_verts)
counts_verts *= 255
counts_verts = 255 - counts_verts.astype(int).reshape(-1,1)
# print("counts_verts", counts_verts.shape)
# print("counts", np.max(counts_verts))

colors = np.hstack([counts_verts, counts_verts, counts_verts, [[255]] * len(counts_verts)])
# print("colors", colors)

model.visual.vertex_colors = colors
model.show()
