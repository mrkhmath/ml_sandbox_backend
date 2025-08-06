import os
import json
import torch
from py2neo import Graph

# === Config ===
STUDENT_SEQ_PATH = "data/student_sequences.json"
SUBGRAPH_DIR = "data/pyg_subgraphs"
ENRICHED_DIR = "data/enriched_subgraphs"
HISTORY_SCORE_PATH = "data/history_scores.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "QQAZqqaz@1"

os.makedirs(ENRICHED_DIR, exist_ok=True)

# === Load all students ===
with open(STUDENT_SEQ_PATH) as f:
    student_sequences = json.load(f)

# === Extract concept scores and required concept codes ===
history_scores = {}  # {student_id: {concept_code: normalized_score}}
required_concepts = set()

for student_id, seq in student_sequences.items():
    history_scores[student_id] = {}
    for entry in seq:
        code = entry.get("canonical_ccss")
        score = entry.get("score")
        if code and score is not None:
            history_scores[student_id][code] = score
            required_concepts.add(code)

# === Connect to Neo4j and fetch metadata ===
print("🔌 Connecting to Neo4j...")
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("📥 Fetching concept metadata...")
concept_metadata = {}
query = """
MATCH (c:Concept)
WHERE c.code IS NOT NULL
RETURN c.code AS code, c.grade_levels AS grade_levels, c.description AS description
"""

for record in graph.run(query):
    concept_metadata[record["code"]] = {
        "grade_levels": record["grade_levels"],
        "description": record["description"]
    }

# === Enrich subgraphs ===
print("🧠 Enriching subgraphs...")
enriched_count = 0
skipped = []

for code in sorted(required_concepts):
    pt_path = os.path.join(SUBGRAPH_DIR, f"{code}.pt")
    if not os.path.exists(pt_path):
        print(f"❌ Missing: {code}.pt — skipping")
        skipped.append(code)
        continue

    data = torch.load(pt_path)

    if not hasattr(data, "code_strs"):
        print(f"❌ No 'code_strs' in {code}.pt — skipping")
        skipped.append(code)
        continue

    grade_levels = []
    descriptions = []
    per_node_scores = []

    for node_code in data.code_strs:
        meta = concept_metadata.get(node_code, {})
        grade_levels.append(meta.get("grade_levels", []))
        descriptions.append(meta.get("description", ""))

        student_score_map = {
            sid: history_scores[sid][node_code]
            for sid in history_scores
            if node_code in history_scores[sid]
        }
        per_node_scores.append(student_score_map)

    data.grade_levels = grade_levels
    data.descriptions = descriptions
    data.history_scores = per_node_scores

    torch.save(data, os.path.join(ENRICHED_DIR, f"{code}.pt"))
    enriched_count += 1

# === Save full history_scores.json ===
with open(HISTORY_SCORE_PATH, "w") as f:
    json.dump(history_scores, f, indent=2)

print(f"\n✅ Enriched {enriched_count} subgraphs.")
if skipped:
    print(f"⚠️ Skipped {len(skipped)} due to missing files or code_strs.")
