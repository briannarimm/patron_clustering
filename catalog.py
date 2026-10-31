"""What can be checked out: formats and subject areas."""

FORMATS = ["book", "dvd", "audiobook", "large_print", "ebook", "periodical"]

# Interest dimensions (the latent interest mix is a distribution over these)
INTERESTS = [
    "adult_fiction",
    "adult_nonfiction",
    "academic_reference",
    "kids_books",
    "kids_media",      # DVDs of shows, etc.
    "teen_ya",
    "adult_media",     # movies, TV box sets
    "hobby_practical", # cooking, crafts, home repair
]

# Format mix for each interest. Rows sum to 1.
INTEREST_TO_FORMAT: dict[str, dict[str, float]] = {
    "adult_fiction":      {"book": 0.60, "audiobook": 0.15, "large_print": 0.10, "ebook": 0.15},
    "adult_nonfiction":   {"book": 0.65, "audiobook": 0.10, "ebook": 0.15, "periodical": 0.10},
    "academic_reference": {"book": 0.70, "ebook": 0.10, "periodical": 0.20},
    "kids_books":         {"book": 0.90, "audiobook": 0.05, "ebook": 0.05},
    "kids_media":         {"dvd": 0.90, "book": 0.10},
    "teen_ya":            {"book": 0.55, "audiobook": 0.15, "ebook": 0.30},
    "adult_media":        {"dvd": 0.95, "periodical": 0.05},
    "hobby_practical":    {"book": 0.70, "dvd": 0.10, "periodical": 0.20},
}

# Subject labels, loosely Dewey for nonfiction. "j" = juvenile, "ya" = young adult.
INTEREST_TO_SUBJECTS: dict[str, list[str]] = {
    "adult_fiction":      ["general fiction", "mystery", "romance", "science fiction",
                           "historical fiction", "thriller"],
    "adult_nonfiction":   ["biography", "history", "self-help", "true crime", "current events"],
    "academic_reference": ["100s philosophy", "300s social science", "400s language",
                           "500s science", "600s technology", "900s history", "reference"],
    "kids_books":         ["j picture book", "j easy reader", "j fiction", "j nonfiction"],
    "kids_media":         ["j tv series", "j movie", "j music"],
    "teen_ya":            ["ya fiction", "ya fantasy", "ya graphic novel", "ya nonfiction"],
    "adult_media":        ["movie", "tv series", "documentary"],
    "hobby_practical":    ["630s gardening", "640s cooking", "690s home repair",
                           "740s crafts", "790s sports"],
}

# Subject groups that features.py counts. These are things a cataloger could
# derive from the call number, not from the latent interest.
KIDS_SUBJECTS = {s for s in sum(INTEREST_TO_SUBJECTS.values(), []) if s.startswith("j ")}
FICTION_SUBJECTS = set(INTEREST_TO_SUBJECTS["adult_fiction"]) | {"j fiction", "ya fiction", "ya fantasy"}
ACADEMIC_SUBJECTS = set(INTEREST_TO_SUBJECTS["academic_reference"])
