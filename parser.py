import csv
import re

def parse_system_doc(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    components = []
    edges = []

    section = None  # None | "components" | "edges"

    # Edge patterns:
    # 1) "- A -> B [relation]"
    # 2) "- A <-> B [relation]"  (for bidirectional)
    arrow_uni_pattern = re.compile(r"^- (.+?) -> (.+?) \[(.+?)\]\s*$")
    arrow_bi_pattern  = re.compile(r"^- (.+?) \<\-\> (.+?) \[(.+?)\]\s*$")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Section headers
        if line.startswith("Components:"):
            section = "components"
            continue
        if line.startswith("Edges:"):
            section = "edges"
            continue

        # Components
        if section == "components" and line.startswith("- "):
            comp_name = line[2:].strip()
            if comp_name:
                components.append(comp_name)
            continue

        # Edges
        if section == "edges" and line.startswith("- "):
            # Try bidirectional with "<->"
            m_bi = arrow_bi_pattern.match(line)
            if m_bi:
                source, target, relation = m_bi.groups()
                relation = relation.strip().lower()

                # For bidirectional, create two unidirectional edges
                edges.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "relation": "unidirectional"
                })
                edges.append({
                    "source": target.strip(),
                    "target": source.strip(),
                    "relation": "unidirectional"
                })
                continue

            # Normal "A -> B [relation]" pattern
            m_uni = arrow_uni_pattern.match(line)
            if not m_uni:
                raise ValueError(f"Cannot parse edge line: {line}")

            source, target, relation = m_uni.groups()
            relation = relation.strip().lower()

            if relation == "bidirectional":
                # Expand to two directed edges
                edges.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "relation": "unidirectional"
                })
                edges.append({
                    "source": target.strip(),
                    "target": source.strip(),
                    "relation": "unidirectional"
                })
            else:
                edges.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "relation": relation
                })

    return components, edges


def write_csvs(components, edges,
               components_path="components.csv",
               edges_path="edges.csv"):
    # Components CSV: name
    with open(components_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for comp in components:
            writer.writerow([comp])

    # Edges CSV: source, target, relation
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "relation"])
        for e in edges:
            writer.writerow([e["source"], e["target"], e["relation"]])


if __name__ == "__main__":
    input_path = "MainSystem.txt"

    components, edges = parse_system_doc(input_path)
    write_csvs(components, edges,
               components_path="components_mainsystem.csv",
               edges_path="edgesMainsystem.csv")

    print("Wrote components.csv and edges.csv")
