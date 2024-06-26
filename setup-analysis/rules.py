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

# import pandas as pd

# stim_mapping = {
#     "alarm-clock": 0,
#     "apps": 1,
#     "baby-carriage": 2,
#     "bell": 3,
#     "biking": 4,
#     "bone": 5,
#     "box-open": 6,
#     "brightness": 7,
#     "broadcast-tower": 8,
#     "bulb": 9,
#     "camera": 10,
#     "candy-cane": 11,
#     "carrot": 12,
#     "chess": 13,
#     "club": 14,
#     "cocktail-alt": 15,
#     "cube": 16,
#     "diamond": 17,
#     "eye": 18,
#     "fish": 19,
#     "gamepad": 20,
#     "gift": 21,
#     "globe": 22,
#     "graduation-cap": 23,
#     "guitar": 24,
#     "hammer": 25,
#     "hand-horns": 26,
#     "headphones": 27,
#     "heart": 28,
#     "helicopter-side": 29,
#     "home": 30,
#     "ice-skate": 31,
#     "island-tropical": 32,
#     "key": 33,
#     "lock": 34,
#     "megaphone": 35,
#     "mug-hot-alt": 36,
#     "music-alt": 37,
#     "paper-plane": 38,
#     "paw": 39,
#     "peach": 40,
#     "phone-call": 41,
#     "plane-alt": 42,
#     "playing-cards": 43,
#     "pyramid": 44,
#     "question-mark": 45,
#     "rocket": 46,
#     "rugby": 47,
#     "search": 48,
#     "settings": 49,
#     "shopping-basket": 50,
#     "shopping_cart": 51,
#     "skiing": 52,
#     "smile": 53,
#     "social-network": 54,
#     "spade": 55,
#     "star": 56,
#     "trophy-star": 57,
#     "truck-side": 58,
#     "user": 59,
#     "wheat": 60,
# }

# stim_mapping = {v: f"{k}.png" for k, v in stim_mapping.items()}
# pd.DataFrame(stim_mapping.items(), columns=["img_id", "img_name"]).to_csv(
#     "images.csv", index=False
# )
