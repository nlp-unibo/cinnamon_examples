# Cinnamon Examples

This project contains a complete example of a machine-learning pipeline.

In particular, we'll see how

- Define a ``DataLoader``
- Define multiple data ``Processor``
- Define a SVM classifier ``Model``
- Train and evaluate our classifier with a ``Pipeline``

We consider the IMDB data to perform a binary classification for sentiment analysis.

## Documentation

Check the online documentation of [cinnamon-examples](https://nlp-unibo.github.io/cinnamon_examples) for a detailed walkthrough on the project.

## Usage

Want to use the ``Component`` and ``Configuration`` of this example project?

You can quickly integrate them into your project by adding this Github repo as follows:
```python
Registry.setup(directory=...,
               external_directories=[
                   ...,
                   "https://github.com/nlp-unibo/cinnamon_examples"
               ])
```

Done! Cinnamon will automatically import all registered ``Configuration`` under `configurations` folders.

You can code your custom ``Configuration`` with children pointing to the ``RegistrationKey`` of this project!

## Contact

Don't hesitate to contact:
- Federico Ruggeri @ [federico.ruggeri6@unibo.it](mailto:federico.ruggeri6@unibo.it)

for questions/doubts/issues!