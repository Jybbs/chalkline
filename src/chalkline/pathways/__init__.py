"""
Career pathway graph construction and traversal.

Bridges the geometry track (HAC clusters from `clustering`) and the
co-occurrence track (PMI edges from `association`) into a directed weighted
career graph with sector, apprenticeship, and educational program
enrichments. The primary graph retains cycles for legitimate lateral
transitions, with a derived DAG view for longest-path visualization.
"""
