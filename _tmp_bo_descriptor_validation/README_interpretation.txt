This temporary validation distinguishes the final 8-feature model from a low-dimensional projection surrogate.
The final 8-feature model does not contain precursor volume or metal fraction inputs, so it cannot directly produce a continuous prediction field over BO composition ratios without additional assumptions.
Best 3-descriptor projection model: extra_trees_3desc. See weighted_descriptor_model_validation.csv for held-out and CV R2.
The projection surrogate is useful only as a qualitative descriptor-map/volcano check, not as the formal predictive model used in the manuscript.
