# TF_pyDF

TF_pyDF is a Python package that provides a module for calculating [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) (Term Frequency-Inverse Document Frequency). It allows you to create a document model and perform various operations such as adding and removing documents, searching for documents based on queries, and computing TF-IDF scores.

## Usage Warning

The document contents and search query have to be pre-tokenized into list[str] or in a Tokenizer Iterator.
You can use your own tokenizer or use one related to this package - [LexiPy](https://pypi.org/project/lexipy/), it provides a simple way to tokenize a string of text into tokens.

## Installation

You can install TF_pyDF using pip:

```shell
pip install tf_pydf
```

## Usage

```python
from tf_pydf import Model

documents = {
    "fruits":
        ["apple", "banana", "orange"],
    "vegetables":
        ["tomato", "cucumber", "radish"],
    "pasta":
        ["tagliatelle", "rotini", "rigatoni"],
    }

# Create a new instance of the document model
model = Model()

# Add documents to the model
for doc_id, doc_content in documents.items():
    model.add_doc(doc_id, doc_content)

# Or use convenience method "from_dict"
model = Model.from_dict(documents)

# Remove a document from the model
doc_id = "pasta"
model.remove_doc(doc_id)

# Check if a document is in the model
if doc_id in model:
    ...

# Search the model for documents matching a query
results = model.search_query(query)

results
>>> [('fruits', 0.10034333188799373), ('vegetables', 0.0)]
```

## Contributing

Contributions to LexiPy are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/t4skmanag3r/tf_pydf).

## License

This package is licensed under the [MIT License](https://opensource.org/licenses/mit).

## References

- [Github Repository](https://github.com/t4skmanag3r/tf_pydf)
- [PyPI package](https://pypi.org/project/tf_pydf/)
- [LexiPy](https://pypi.org/project/lexipy/) - tokenizer
- [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) - wikipedia
