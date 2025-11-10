---
title: "2D Delaunay Triangulations"
comments: true
---
# 2D Delaunay Triangulations
!!! quote "Related Works"
    **[Delaunay Mesh Generation](https://people.eecs.berkeley.edu/~jrs/meshbook.html)**

## 1. Introduction

In two dimensions, the Delaunay triangulations has a striking advantage: among all possible triangulations of a fixed set of points, the Delaunay triangulations maximizes the minimum angle. It also optimizes serveral other geometric criteria related to interpolation accuracy.

A constrained triangulation is a triangulation that enforces the presence of specified edges -- for example, the boundary of a nonconvex object. A constrained Delaunay triangulation relaxes the Delaunay property just enough to recover those edges, while enjoying optimality properties similar to those of a Delaunay triangulation.

## 2. The Delaunay triangulation

The **Delaunay triangulation** of a point set $S$, introduced by Boris Nikolaevich Delaunay in 1934, is characterized by the *empty circumdisk property*: no point in $S$ lies in the interior of any triangle's circumscribing disk.

!!! Abstract "Definition 2.2(Delaunay)"
    In the context of a finite point set $S$, a triangle is *Delaunay* if its vertices are in $S$ and its open circumdisk is *empty* -- i.e. contains no point in S. Note that any number of points in $S$ can lie on a Delaunay triangle's circumcircle. An edge is *Delaunay* if its vertices are in $S$ and it has at least one empty open circumdisk. A *Delaunay triangulation* of $S$, denoted $\text{Del}\, S$,