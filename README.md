# CS229B Final Project
Hierarchical load forecasting \
Ines Dormoy and Louis Gautier 

Forecasting energy load at several scales is a crucial task for grid managers and
utilities, particularly with the rise of intermittent renewable energy sources demanding a heightened level of planning. In this project, we explore ways to predict
energy demand at several levels of hierarchy in a grid, from individual meters
to large substations serving entire neighborhoods. A desirable property in this
context is that the sum of the load time series at the bottom of the hierarchy (e.g.,
individual meters) matches the load at higher levels. We propose two methods to
achieve this hierarchical load forecasting objective: a two-step approach consisting
of reconciling base learners and a deep learning-based end-to-end pipeline. We
benchmark several design choices for these two methods and analyze whether they
improve accuracy at various levels of the hierarchy. Finally, we compare them in
terms of robustness and tractability on real massive datasets.
