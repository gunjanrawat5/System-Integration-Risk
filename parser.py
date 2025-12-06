import json
import re

def parse_system_doc(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    components = []
    edges = []

    section = None  # None | "components" | "edges"

    # Regex for edge lines: "- A -> B [relation]"
    edge_pattern = re.compile(r"^- (.+?) -> (.+?) \[(.+?)\]\s*$")

    for raw in lines:
        line = raw.strip()

        if not line:
            continue  # skip empty lines

        # Detect sections
        if line.startswith("Components:"):
            section = "components"
            continue
        if line.startswith("Edges:"):
            section = "edges"
            continue

        # Parse components
        if section == "components" and line.startswith("- "):
            comp_name = line[2:].strip()
            if comp_name:
                components.append(comp_name)
            continue

        # Parse edges
        if section == "edges" and line.startswith("- "):
            m = edge_pattern.match(line)
            if not m:
                raise ValueError(f"Cannot parse edge line: {line}")
            source, target, relation = m.groups()
            edges.append({
                "source": source.strip(),
                "target": target.strip(),
                "relation": relation.strip()
            })

    return {
        "components": components,
        "edges": edges
    }


if __name__ == "__main__":
    # Change this to your actual filename
    input_path = "payment_gateway_system.txt"

    system_json = parse_system_doc(input_path)

    # Pretty-print to stdout
    print(json.dumps(system_json, indent=2, ensure_ascii=False))
