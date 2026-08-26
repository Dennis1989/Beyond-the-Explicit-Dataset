#!/usr/bin/env python
# coding: utf-8

"""
Generate Dehumanization Combinations
This script generates all combinations of keywords and descriptors and saves them to a CSV file.
"""

import pandas as pd
import itertools
from pathlib import Path

# Dehumanization descriptors
EXPLICIT_ANIMALISTIC = [
    "subhumans", "animals", "beasts", "brute", "cattle", "cows", "dogs", "herds", "hog", "horses", "livestock",
    "pigs", "sheep", "rats", "worms", "bugs", "parasites", "insects", "the pest", "flea", "rodents",
    "a virus", "a disease", "ants", "beast-like", "bedbugs", "brainworms", "breeders", "broodstock",
    "buggs", "chimps", "conker", "drat", "earthworms", "entities", "ferals", "ferocious", "ferrets",
    "fester", "fleabags", "garanimal", "hampster", "lice", "maggots", "mice", "ziopig"
]

EXPLICIT_MECHANISTIC = [
    "objects", "automatons", "machines", "machinery", "equipment", "apparatus", "appliances",
    "hardware", "engines", "tools", "devices", "robots", "trash", "garbage", "filth"
]

IMPLICIT_ANIMALISTIC = [
    "are uncultured", "are unrefined", "are provincial", "are backward", "are uncivilised", "are uncivilized",
    "are coarse", "are unintelligent", "are illiterate", "are rude", "have no manners",
    "lack morals", "corrupt", "moralless", "without morals", "are prone to violating the moral code",
    "are irrational", "are childish in their decision-making", "are childlike", "are immature",
    "are naive", "are immoral", "are greedy", "are selfish", "are uneducated"
]

IMPLICIT_MECHANISTIC = [
    "have no emotion", "lack emotion", "are cold", "are unemotional", "are detached", "are apathetic",
    "are distant", "are inactive in their lives", "are inactive", "are generally unresponsive",
    "are set in their ways", "do not change their mind", "are passive", "are interchangeable",
    "are fungible", "are shallow", "lack depth", "are simplistic", "are inert", "are superficial",
    "are unfeeling", "are inflexible", "are one dimensional", "are lethargic", "are sluggish", "are narrow-minded", "are robotic"
]

PLURAL_KEYWORDS = [
    "immigrants", "refugees", "aliens", "illegals", "escapees", "foreigners", "displaced persons",
    "undesirables", "migrants", "jews", "hebrews", "kikes", "yids", "mockies", "mockys", "bootlips",
    "dindus", "porch monkeys", "ghetto monkeys", "golliwogs", "hoodrats", "blacks", "black people",
    "jiggas", "jiggers", "jigros", "kneegrows", "moulies", "negros", "niggers", "afro-americans",
    "afromericans", "native-americans", "women", "girls", "bitches", "cunts", "dykes", "pussies",
    "hoes", "bints", "females", "shemales", "transexuals", "twats", "asexuals", "faggots",
    "homosexuals", "americans", "asians", "turks", "arabs", "chinese", "mexicans", "africans",
    "indians", "amerindians", "italians", "spanish", "russians", "scots", "french", "syrians",
    "dutch", "ukrainians", "republicans", "democrats", "dems", "reps", "cripples", "gimps",
    "libtards", "mongos", "retards", "tarded", "tards", "elderlies", "oldppl", "oldwomen",
    "old persons", "patients", "criminals", "inmates", "prisoners", "you", "ukrainians",
    "israelis", "palestinians", "they", "those", "these", "workers", "9-to-5ers", "toursits",
    "travellers", "businessmen"
]

def generate_combinations_dataframe() -> pd.DataFrame:
    """
    Generate all combinations of keywords and descriptors and return as a DataFrame.
    
    Returns:
        DataFrame with columns: type, keyword, descriptor, combination
    """
    rows = []
    
    # Explicit mechanistic: "{kw} are like {desc}"
    for kw, desc in itertools.product(PLURAL_KEYWORDS, EXPLICIT_MECHANISTIC):
        combination = f"{kw} are like {desc}"
        rows.append({
            "type": "explicit_mechanistic",
            "keyword": kw,
            "descriptor": desc,
            "combination": combination
        })
    
    # Implicit mechanistic: "{kw} {desc}"
    for kw, desc in itertools.product(PLURAL_KEYWORDS, IMPLICIT_MECHANISTIC):
        combination = f"{kw} {desc}"
        rows.append({
            "type": "implicit_mechanistic",
            "keyword": kw,
            "descriptor": desc,
            "combination": combination
        })
    
    # Explicit animalistic: "{kw} are like {desc}"
    for kw, desc in itertools.product(PLURAL_KEYWORDS, EXPLICIT_ANIMALISTIC):
        combination = f"{kw} are like {desc}"
        rows.append({
            "type": "explicit_animalistic",
            "keyword": kw,
            "descriptor": desc,
            "combination": combination
        })
    
    # Implicit animalistic: "{kw} {desc}"
    for kw, desc in itertools.product(PLURAL_KEYWORDS, IMPLICIT_ANIMALISTIC):
        combination = f"{kw} {desc}"
        rows.append({
            "type": "implicit_animalistic",
            "keyword": kw,
            "descriptor": desc,
            "combination": combination
        })
    
    return pd.DataFrame(rows)

def main() -> None:
    """Main execution function."""
    print("Generating combinations...")
    df = generate_combinations_dataframe()
    
    print(f"Generated {len(df)} combinations")
    print(f"Breakdown by type:")
    print(df["type"].value_counts())
    
    # Save to CSV
    output_file = Path("../data/german_dehumanization_templates.csv")
    df.to_csv(output_file, index=False, sep="\t")
    print(f"\nCombinations saved to: {output_file}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()



