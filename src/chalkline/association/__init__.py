"""
PMI co-occurrence network and Apriori comparison.

Constructs a PMI-weighted skill co-occurrence graph from the binary
matrix produced by `extraction`, applies Louvain community detection
for career track grouping, and compares against Apriori association
rules for the DS5230 deliverable. The Apriori module is excluded from
the production pipeline.
"""
