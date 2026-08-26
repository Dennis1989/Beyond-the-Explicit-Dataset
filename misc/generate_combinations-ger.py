#!/usr/bin/env python
# coding: utf-8

"""
Generate Dehumanization Combinations (German)
This script generates all combinations of keywords and descriptors and saves them to a CSV file.
"""

import pandas as pd
import itertools
from pathlib import Path

# Dehumanization keywords
EXPLICIT_ANIMALISTIC = [
    "Untermenschen", "Tiere", "Bestien", "Bestie", "Vieh", "Kühe", "Hunde",
    "Herden", "Schwein", "Pferde", "Nutztiere", "Schweine", "Schafe", "Ratten",
    "Würmer", "Ungeziefer", "Parasiten", "Insekten", "die Pest", "Floh",
    "Nagetiere", "ein Virus", "eine Krankheit", "Ameisen", "bestienhaft",
    "Bettwanzen", "Gehirnwürmer", "Züchter", "Zuchtbestand", "Käfer",
    "Schimpansen", "Kastanie", "verflucht", "Regenwürmer", "Entitäten",
    "verwilderte Tiere", "wild", "Frettchen", "Eiter", "Flohsäcke",
    "garanimal", "Hamster", "Läuse", "Maden", "Mäuse", "ziopig"
]

EXPLICIT_MECHANISTIC = [
    "Objekte", "Gegenstände", "Automaten", "Maschinen", "Maschinerie",
    "Ausrüstung", "Apparat", "Geräte", "Hardware", "Motoren", "Werkzeuge",
    "Vorrichtungen", "Roboter", "Müll", "Abfall", "Dreck"
]

IMPLICIT_ANIMALISTIC = [
    "sind unkultiviert", "sind ungehobelt", "sind provinziell",
    "sind rückständig", "sind unzivilisiert", "sind grob",
    "sind unintelligent", "sind Analphabeten", "sind unhöflich",
    "haben keine Manieren", "haben keine Moral", "korrupt",
    "ohne Moral", "sind anfällig dafür, den Moralkodex zu verletzen",
    "sind irrational", "verhalten sich kindisch in ihrer Entscheidungsfindung",
    "sind kindlich", "sind unreif", "sind naiv", "sind unmoralisch",
    "sind gierig", "sind egoistisch", "sind ungebildet"
]

IMPLICIT_MECHANISTIC = [
    "haben keine Emotionen", "fehlt es an Emotionen", "sind kalt",
    "sind emotionslos", "sind distanziert", "sind apathisch",
    "sind abweisend", "sind in ihrem Leben inaktiv", "sind inaktiv",
    "sind allgemein nicht ansprechbar", "sind festgefahren in ihrer Denkweise",
    "ändern ihre Meinung nicht", "sind passiv", "sind austauschbar",
    "sind fungibel", "sind oberflächlich", "ihnen fehlt es an Tiefe",
    "sind simpel", "sind träge", "sind gefühllos", "sind unflexibel",
    "sind eindimensional", "sind lethargisch", "sind engstirnig",
    "sind robotisch"
]

PLURAL_KEYWORDS = [
    "Flüchtlinge", "Immigranten", "Ausländer", "Vertriebene", "Unerwünschte",
    "Migranten", "Juden", "Jews", "Schwarze", "Neger", "Niggers",
    "Afroamerikaner", "Ureinwohner Amerikas", "Frauen", "Mädchen", "Schlampen",
    "Fotzen", "Lesben", "Muschis", "Huren", "Tussis", "Weibliche", "Transen",
    "Transsexuelle", "Asexuelle", "Schwuchteln", "Homosexuelle", "Schwule",
    "Schwule Menschen", "Deutsche", "Asiaten", "Türken", "Araber", "Mexikaner",
    "Afrikaner", "Amerindianer", "Russen", "Schotten", "Franzosen",
    "Niederländer", "Ukrainer", "Konservative", "Demokraten", "Behinderte",
    "Libtards", "Mongos", "Spasten", "Tards", "Ältere", "Ältere Frauen",
    "Ältere Männer", "Senioren", "Alte Leute", "Alte Personen", "Patienten",
    "Kriminelle", "Diebe", "Straftäter", "Häftlinge", "Gefangene", "Israelis",
    "Palästinenser", "Arbeiter", "Reisende", "Geschäftsmänner", "Grüne",
    "Ökos Grünlinge", "Sozialdemokraten", "SPDler", "CDUler", "Christdemokraten",
    "CSUler", "AfDler", "Blaue", "Rechtskonservative", "Liberale", "FDPler",
    "Gelbe", "Linke"
]

def generate_combinations_dataframe() -> pd.DataFrame:
    """
    Generate all combinations of keywords and descriptors and return as a DataFrame.
    
    Returns:
        DataFrame with columns: type, keyword, descriptor, combination
    """
    rows = []
    
    # Explicit animalistic: "{keyword.lower()} sind wie {phrase.lower()}"
    for keyword, phrase in itertools.product(PLURAL_KEYWORDS, EXPLICIT_ANIMALISTIC):
        combination = f"{keyword.lower()} sind wie {phrase.lower()}"
        rows.append({
            "type": "explicit_animalistic",
            "keyword": keyword,
            "descriptor": phrase,
            "combination": combination
        })
    
    # Implicit animalistic: "{keyword.lower()} {phrase.lower()}"
    for keyword, phrase in itertools.product(PLURAL_KEYWORDS, IMPLICIT_ANIMALISTIC):
        combination = f"{keyword.lower()} {phrase.lower()}"
        rows.append({
            "type": "implicit_animalistic",
            "keyword": keyword,
            "descriptor": phrase,
            "combination": combination
        })
    
    # Explicit mechanistic: "{keyword.lower()} sind wie {phrase.lower()}"
    for keyword, phrase in itertools.product(PLURAL_KEYWORDS, EXPLICIT_MECHANISTIC):
        combination = f"{keyword.lower()} sind wie {phrase.lower()}"
        rows.append({
            "type": "explicit_mechanistic",
            "keyword": keyword,
            "descriptor": phrase,
            "combination": combination
        })
    
    # Implicit mechanistic: "{keyword.lower()} {phrase.lower()}"
    for keyword, phrase in itertools.product(PLURAL_KEYWORDS, IMPLICIT_MECHANISTIC):
        combination = f"{keyword.lower()} {phrase.lower()}"
        rows.append({
            "type": "implicit_mechanistic",
            "keyword": keyword,
            "descriptor": phrase,
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



