# Abstract
Open set recognition (OSR) is a well-studied problem that tasks machine learning clas-
sifiers with labeling both known and unknown instances. Most existing OSR methods
rely on probability- or distance-based thresholds to classify unknown instances, largely
restricting the potential for understanding how these instances may relate to known
classes. This paper introduces a novel OSR algorithm that addresses this limitation
by using a continuous abatement model to adjust predicted class probabilities based
on similarity in feature space. The algorithm returns a set of |K| + 1 confidence values
for any test instance, where |K| is the number of known classes included in training,
and the open set is treated as the |K| + 1st class. We empirically demonstrate that
these confidence values enable accurate classification and interpretable contextualiza-
tion across known and unknown instances. We evaluate our approach on seven datasets,
including one involving the classification of integer programming instances into com-
binatorial problem classes (e.g., bin packing, lot sizing). Across these experiments,
our algorithm frequently outperforms state-of-the-art OSR methods while, for the first
time, enabling contextualization of unfamiliar inputs with respect to known classes
– an advantage with relevance in domains such as optimization, medical diagnostics,
autonomous driving, and defect detection, among others.
