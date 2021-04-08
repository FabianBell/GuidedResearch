# Guided Research - Modifying the Writing Style in Goal-Oriented Dialog Generation

## Master
This branch contains the report as well as the first tries regarding the textsettr and soloist approach.
## Soloist
This branch contains the code for the individual Soloist approach. Based on this code the reported reseults were generated.
## Textsettr
The textsettr branch contains the code for the different textsettr approaches. The HEAD contains the small individual textsettr approach. For training the large version, the model name 't5-small' has to be replaced to 't5-large' in every occurance. However, this requires parallelisation (by using the `parallelize` function) on at least 4 gpus with 16 GB RAM. The merged version can be used as the small version however has to be trained according to the report. 
## DGST
The dgst branch contains the code for the dgst approach.
## Classifier
The classifier branch contains the training of the classifier used for evaluation.
