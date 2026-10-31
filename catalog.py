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

# TODO: map each interest to a distribution over (format, subject) so events
# carry realistic format/subject values. Example:
#   "kids_media"    -> {"dvd": 0.9, "book": 0.1}
#   "adult_fiction" -> {"book": 0.6, "audiobook": 0.2, "large_print": 0.1, "ebook": 0.1}
INTEREST_TO_FORMAT: dict[str, dict[str, float]] = {}

# TODO: map each interest to a set of subject/Dewey labels, e.g.
#   "academic_reference" -> ["300s social science", "500s science", ...]
INTEREST_TO_SUBJECTS: dict[str, list[str]] = {}
