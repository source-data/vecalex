# vecalex

Augment OpenAlex entities with vector representations computed from their associated works' abstracts.

## Usage

Basic example: where should I post an artificial intelligence preprint?

```python
from vecalex import config, Sources, Subfields
from sklearn.metrics.pairwise import cosine_similarity

ai_subfields = Subfields().search("artificial intelligence").get()  # fetch subfields with AI in their name
repositories = [  # fetch the 25 most cited preprint repositories
    source
    for page in Sources().filter(type="repository").sort(cited_by_count="desc").paginate(n_max=25)
    for source in page
]

subfield_repo_similarities = cosine_similarity(
    [s["vec"] for s in ai_subfields],
    [r["vec"] for r in repositories],
)
# print top 3 most similar repositories for each subfield
for i, subfield in enumerate(ai_subfields):
    print(f"Top repositories for subfield {subfield['display_name']}:")
    top3_indices = subfield_repo_similarities[i].argsort()[-3:][::-1]
    for rank, idx in enumerate(top3_indices, start=1):
        repo = repositories[idx]
        sim = subfield_repo_similarities[i][idx]
        print(f"  {rank}. {repo['display_name']} (similarity: {sim:.4f})")
# Sample output:
# Top repositories for subfield Artificial Intelligence:
#   1. arXiv (Cornell University) (similarity: 0.8439)
#   2. OPAL (Open@LaTrobe) (La Trobe University) (similarity: 0.5735)
#   3. Munich Personal RePEc Archive (Ludwig Maximilian University of Munich) (similarity: 0.5101)
```
