# Anomaly-Detection---Feature-Selection
Using Different Methods for Feature Selection such as Filter, Wrapper and embedded Methods to extract the features of interest in the breast cancer dataset.
Through this Project, I have applied 2 popular Feature Selection Methods.
I have tried the Filter method and Wrapper method.
I couldn't use the embbeded method as it can only applied to linear(numeric) data and I have a categorical classification(classification)problem.

When Filter method was used, I got 21 features selected at the end.
where as using the wrapper method I could use only 15 features that could result in high accuracy as well.

Other benefits of using wrapper method is it adjusts the selected features based the accuracy of the model, unlike the filter method, it selects the features without the need of a model.

Filter methods reported high accuracy(above 90) but Wrapper method reported higher accuracies (above 95 for most models)
