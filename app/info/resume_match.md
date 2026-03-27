Your resume was encoded with the same **sentence transformer** used on all 922 job postings, then projected into the pipeline's 10-dimensional SVD space (*the same reduction applied during clustering*). The match is determined by **nearest-centroid classification**, finding which of the 20 career family centroids your projected resume vector is closest to via Euclidean distance.

The distances to all centroids produce the proximity rankings shown below, and the **matched career family** is the one with the smallest distance to your resume's position in the embedding space.
