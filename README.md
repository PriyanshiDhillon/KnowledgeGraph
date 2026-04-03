# Modeling Human-AI Collaboration Dynamics: A Knowledge Graph Approach

**Authors:** Gergely Bela Banyai (2894545), Priyanshi Dhillon (2879065)  
**Course:** Knowledge Graphs and Semantic Technologies  
**Institution:** Vrije Universiteit Amsterdam  
**Date:** April 2026

---

## Overview

This project constructs and analyzes a Knowledge Graph (KG) of Human-AI collaboration dynamics, focusing on how **autonomy level**, **explainability level**, and **agreement level** jointly influence **outcome quality** across different collaboration scenarios.

We extend the [Hybrid Intelligence Ontology v2](https://github.com/HybridIntelligence/HI-ontology) with four analytical dimensions and populate it with:
- **21 use cases** extracted from 7 HHAI-domain research papers
- **6 Human-AI collaboration scenarios** (3 from real HI competition data, 3 manually designed intermediate cases)

The graph is analyzed using SPARQL queries, graph metrics (density, PageRank, betweenness centrality), and a ComplEx embedding model for link prediction via [PyKEEN](https://pykeen.readthedocs.io/).

**Key finding:** High explainability is a stronger predictor of good outcomes than high autonomy alone. Forced agreement always leads to bad outcomes regardless of other factors.

---

## Repository Structure

```
KnowledgeGraph/
│
├── Final.ttl                  # Extended HI Ontology (TBox + ABox) in Turtle format
├── pg_project_ML.ipynb        # Jupyter notebook: KG metrics + ComplEx link prediction
├── report.pdf                 # Full project report (LNCS format)
└── README.md                  # This file
```

---

## Requirements

### For Ontology and SPARQL (GraphDB)

- [GraphDB Free 10.x](https://www.ontotext.com/products/graphdb/download/) — used to load the ontology and run SPARQL queries
- [Protégé 5.x](https://protege.stanford.edu/) — used for ontology development and HermiT reasoning (optional, for inspection)

### For Machine Learning (Python)

Python 3.11 is recommended. Install dependencies with:

```bash
pip install pykeen torch networkx matplotlib rdflib jupyter
```

Or install from a requirements file if provided:

```bash
pip install -r requirements.txt
```

Key packages used:
| Package | Purpose |
|---|---|
| `pykeen` | ComplEx KG embedding and link prediction |
| `torch` | Backend for PyKEEN |
| `networkx` | Graph metrics (density, centrality, clustering) |
| `matplotlib` | Visualizations (degree distribution, PageRank) |
| `rdflib` | Parsing the TTL file for ML preprocessing |

---

## How to Reproduce

### Step 1 — Load the Ontology into GraphDB

1. Download and launch [GraphDB Free](https://www.ontotext.com/products/graphdb/download/)
2. Create a new repository (e.g., `hi-kg`)
3. Import `Final.ttl` via **Import → RDF Files → Upload**
4. Enable the HermiT reasoner: **Repository Settings → Reasoner → OWL2-RL** (or use Protégé for full OWL2 reasoning)

After import, the graph contains **3004 triples** (including inferred statements).

---

### Step 2 — Run the SPARQL Queries

Open the SPARQL editor in GraphDB (`http://localhost:7200`) and run the queries below. All queries use the prefix:

```sparql
PREFIX hi: <http://www.semanticweb.org/hi-ontology#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
```

---

#### Query 1 — Autonomy × Explainability vs Outcome

```sparql
SELECT ?interaction ?autonomy ?explainability ?outcome
WHERE {
  ?interaction rdf:type hi:Interaction .
  ?interaction hi:hasAutonomyLevel ?autonomy .
  ?interaction hi:hasExplainabilityLevel ?explainability .
  ?interaction hi:hasOutcomeQuality ?outcome .
}
```

*Shows which combinations of autonomy and explainability are associated with each outcome type.*

---

#### Query 2 — Graph Visualization of Failure Cases (CONSTRUCT)

```sparql
CONSTRUCT {
  ?interaction hi:hasAutonomyLevel ?autonomy .
  ?interaction hi:hasExplainabilityLevel ?explainability .
  ?interaction hi:hasAgreementLevel ?agreement .
  ?interaction hi:hasOutcomeQuality hi:BadOutcome .
}
WHERE {
  ?interaction rdf:type hi:Interaction .
  ?interaction hi:hasOutcomeQuality hi:BadOutcome .
  ?interaction hi:hasAutonomyLevel ?autonomy .
  ?interaction hi:hasExplainabilityLevel ?explainability .
  ?interaction hi:hasAgreementLevel ?agreement .
}
```

*Generates a subgraph of only the failure scenarios. Render using GraphDB's Visual Graph view.*

---

#### Query 3 — Agreement Level vs Outcome Distribution

```sparql
SELECT ?agreement ?outcome (COUNT(?interaction) AS ?count)
WHERE {
  ?interaction rdf:type hi:Interaction .
  ?interaction hi:hasAgreementLevel ?agreement .
  ?interaction hi:hasOutcomeQuality ?outcome .
}
GROUP BY ?agreement ?outcome
ORDER BY DESC(?count)
```

*Shows how each type of agreement (Aligned / Conflicting / Forced) maps to outcome quality.*

---

#### Query 4 — Team Composition Across Use Cases

```sparql
SELECT ?useCase ?domain ?agentType (COUNT(DISTINCT ?member) AS ?teamSize)
WHERE {
  ?useCase rdf:type hi:UseCase ;
           hi:hasDomain ?domain ;
           hi:hasHITeam ?team .
  ?team hi:hasMember ?member .
  OPTIONAL { ?member rdf:type hi:HumanAgent .
             BIND("HumanAgent" AS ?agentType) }
  OPTIONAL { ?member rdf:type hi:ArtificialAgent .
             BIND("ArtificialAgent" AS ?agentType) }
  FILTER EXISTS {
    ?team hi:hasMember ?aiMember .
    ?aiMember rdf:type hi:ArtificialAgent .
  }
}
GROUP BY ?useCase ?domain ?agentType
ORDER BY ?useCase ?agentType
LIMIT 10
```

*Breaks down team composition (human vs. AI agents) across different use case domains.*

---

### Step 3 — Run the Machine Learning Notebook

```bash
jupyter notebook pg_project_ML.ipynb
```

The notebook covers:
1. **Preprocessing** — parsing `Final.ttl` with RDFLib, removing schema-level predicates (`rdf:type`, `rdfs:label`, `owl:sameAs`, `rdfs:comment`, `rdfs:range`, `rdfs:domain`)
2. **Graph Metrics** — computing density, degree distribution, clustering coefficient, betweenness centrality, and PageRank using NetworkX
3. **ComplEx Training** — training a ComplEx embedding model with PyKEEN across multiple configurations (varying epochs and learning rates)
4. **Evaluation** — reporting MRR, Hits@3, Hits@10 against a random baseline; qualitative link prediction analysis on 19 selected triples

> **Note:** A fixed random seed is used throughout for reproducibility. Training on CPU takes approximately 5–10 minutes for 300 epochs.

---

## Ontology Design Summary

The original HI Ontology v2 was extended with:

| Addition | Type | Purpose |
|---|---|---|
| `AutonomyLevel` (High/Medium/Low) | Class hierarchy | Annotate AI autonomy per interaction |
| `ExplainabilityLevel` (High/Partial/Low) | Class hierarchy | Annotate AI transparency per interaction |
| `AgreementLevel` (Aligned/Conflicting/Forced) | Class hierarchy | Annotate human-AI agreement per interaction |
| `OutcomeQuality` (Good/Mixed/Bad) | Class hierarchy | Annotate interaction outcome |
| `hasAutonomyLevel` etc. | Object properties | Link Interaction to dimension values |
| `CollaborativeAgent` | Union class | Union of HumanAgent and ArtificialAgent |
| `hasOutcomeQuality` | Functional property | Enforces single outcome per interaction |
| `allowsTask` | Inverse property | Inverse of `requiresCapability` |
| Property chain | Inference rule | `performsExecution ∘ realizesTask → executesTask` |
| `Recommendation`, `Explanation`, `Decision` | Process classes | Model AI decision-making flow |

**Reasoning:** HermiT (via Protégé or GraphDB) increases the triple count from 1695 → 3004 (~77% increase) through entailment over property chains, inverse relations, and class hierarchies.

---

## External Links

The KG is connected to the following external vocabularies:

| Vocabulary | Link Type | What is linked |
|---|---|---|
| [PROV-O](https://www.w3.org/TR/prov-o/) | `owl:equivalentClass` | `hi:Agent` ≡ `prov:Agent` |
| [FOAF](http://xmlns.com/foaf/spec/) | `rdfs:subClassOf` | `hi:HumanAgent` ⊆ `foaf:Person` |
| [Schema.org](https://schema.org/) | `rdfs:subClassOf` | Dimension classes ⊆ `schema:PropertyValue` |
| [DBpedia](https://dbpedia.org/) | `owl:sameAs` | 117+ instance-level links |
| [Wikidata](https://wikidata.org/) | `owl:sameAs` | 20 instance-level links |

---

## Key Results

| Metric | Value |
|---|---|
| Total entities | 626 named individuals |
| Total triples (after reasoning) | 3,004 |
| Graph nodes (after filtering) | 547 |
| Graph edges (after filtering) | 979 |
| Graph density | 0.0033 |
| Avg. clustering coefficient | 0.080 |
| ComplEx Hits@10 | 0.0278 (baseline: 0.0194) |

**Main finding:** High explainability consistently predicts good outcomes. Forced agreement exclusively predicts bad outcomes. The ComplEx model violates OWL disjointness constraints (e.g., assigning probability 1.0 to mutually exclusive outcomes simultaneously), motivating the constraint-aware training approach proposed in the paper.

---

## Research Proposal (Summary)

Based on the observed ComplEx failures, we propose **constraint-aware ComplEx training** — integrating OWL disjointness axioms, functional property declarations, and cardinality restrictions as differentiable penalty terms in the training loss:

```
Total Loss = Normal Training Loss + (λ × Constraint Violation Penalty)
```

Constraints are extracted automatically via SPARQL queries over the ontology, making the approach generalizable to any OWL-annotated KG. Evaluation uses MRR/Hits@10 for accuracy and a novel **Constraint Violation Rate (CVR)** metric for logical correctness, verified via HermiT.

---

## Citation

If you use this ontology or dataset, please cite:

```
Banyai, G. B., & Dhillon, P. (2026). Modeling Human-AI Collaboration Dynamics: 
A Knowledge Graph Approach to Autonomy, Explainability, and Outcome Analysis. 
Vrije Universiteit Amsterdam.
```

---

## Contact

- Gergely Bela Banyai — g.b.banyai@student.vu.nl  
- Priyanshi Dhillon — p.dhillon@student.vu.nl
