# BSCG-core

BSCG-core provides a core implementation of the **Behavioural Structure-Aware Cross-Granularity (BSCG)** model for milk-yield prediction using behavioural sensor sequences, recent milk records and available animal-state information.

BSCG preserves the within-day order of sensor observations through two levels of temporal encoding. A local encoder represents observations within individual segments, while a global encoder captures relationships between segments across the input day. The resulting sensor representation is combined with production history and animal-state variables to predict milk yield.

The repository contains the BSCG architecture and supporting tools for input preparation, data splitting, model training and performance evaluation. A general input interface allows users to adapt their own data to the model.

This implementation follows the method described in *Preserving within-day order in behavioural sensor data for milk-yield prediction*. 
