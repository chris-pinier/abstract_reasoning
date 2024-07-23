import json

rules_str = """
AAAA AAA A
AAAB AAA B
AABB AAB B
ABBB ABB B
AABC AAB C
ABCC ABC C
ABBC ABB C
ABAB ABA B
ABBA ABB A
ABCA ABC A
ABAC ABA C
ABAB CDC D
AABB CCD D
ABCD ABC D
ABBA CDD C
ABCD DCB A
ABCD EAB C
ABCD EFA B
ABCD EFG H
ABCD EED C
ABC CBA A B
"""

set([len(i.replace(" ", "")) for i in rules_str.strip("\n").split("\n")])

removed = """
ABAC BCB A
"""

sorted(rules_str.strip("\n").split("\n"))

rules_fmt1 = [[r[:-1].strip(), r[-1]] for r in rules_str.strip("\n").split("\n")]
sorted(rules_fmt1, key=lambda x: x[0])

# print("\n".join([" ".join(rule) for rule in rules]))

with open("config/rules.json", "w") as f:
    json.dump(rules_fmt1, f, indent=4, sort_keys=True)


rules_fmt2 = [r.replace(" ", "") for r in rules_str.strip("\n").split("\n")]

with open("config/rules2.json", "w") as f:
    json.dump(rules_fmt2, f, indent=4, sort_keys=True)


# ! #####################################################################
# ! #####################################################################
