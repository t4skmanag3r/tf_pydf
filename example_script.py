from tf_pydf import Model

documents = {
    "fruits": ["apple", "banana", "orange"],
    "vegetables": ["tomato", "cucumber", "radish"],
    "pasta": ["tagliatelle", "rotini", "rigatoni"],
}

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
query = ["apple"]
results = model.search_query(query)
print(results)
