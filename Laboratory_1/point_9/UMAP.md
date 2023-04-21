# UMAP (Uniform Manifold Approximation and Projection)

## Definition

UMAP is a novel manifold learning technique for dimension reduction. UMA Pisconstructed from a theoretical framework based in Riemannian geometry and algebraic topology. Œe result is a practical scalable algorithm that is applicable to real world data. UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance.

## How UMAP works?

UMAP builds something called a "fuzzy simplicial complex". This is really just a representation of a weighted graph, with edge weights representing the likelihood that two points are connected. To determine connectedness, UMAP extends a radius outwards from each point, connecting points when those radii overlap. Choosing this radius is critical - too small a choice will lead to small, isolated clusters, while too large a choice will connect everything together. UMAP overcomes this challenge by choosing a radius locally, based on the distance to each point's nth nearest neighbor. UMAP then makes the graph "fuzzy" by decreasing the likelihood of connection as the radius grows. Finally, by stipulating that each point must be connected to at least its closest neighbor, UMAP ensures that local structure is preserved in balance with global structure.

Once the high-dimensional graph is constructed, UMAP optimizes the layout of a low-dimensional analogue to be as similar as possible. This process is essentially the same as in t-SNE, but using a few clever tricks to speed up the process.

Summarize:

1. First, the distance between all data points in high-dimensional space is calculated.

2. Then, a graph is constructed in which each data point is a node and the edges of the graph connect nearby data points to each other.

3. Next, an optimization algorithm is used to minimize a cost function that measures the distortion of mapping from high to low dimension. This is done iteratively until an optimal mapping is found.
 
4. Once the optimal mapping is in place, the data can be visualized in the least dimensional space and traditional clustering techniques can be used to group the data into clusters.


## Chararteristics
UMAP is fast, scaling well in terms of both dataset size and dimensionality. For example, UMAP can project the 784-dimensional, 70,000-point MNIST dataset in less than 3 minutes, compared to 45 minutes for scikit-learn's t-SNE implementation. Additionally, UMAP tends to better preserve the global structure of the data. This can be attributed to UMAP's strong theoretical foundations, which allow the algorithm to better strike a balance between emphasizing local versus global structure.

References:
- https://pair-code.github.io/understanding-umap/
- https://arxiv.org/abs/1802.03426
- https://taniwa.es/blog/umap/


