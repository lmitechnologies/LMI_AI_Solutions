"""Identity policy for pose classes and keypoints.

A pose class or keypoint has a permanent id and a separate display name. Everything that stores or compares a
layout carries the id, so renaming a class in Factory invalidates nothing that AIS wrote.

Ids are derived from source names rather than allocated: a COCO category ``person`` and a YOLO class ``person``,
converted months apart, must arrive at the same id without consulting each other. Derivation is therefore a pure
function of the source name, and it is a persisted contract -- this module is a copy of ``pose_identifiers.py``
in the GoFactory Python SDK and must agree with it, and with the TypeScript SDK's ``pose-identifiers.ts``, for
every input. Change it in all three or in none.
"""

import hashlib
import re
import unicodedata
from typing import List

# The longest an id may be, in characters, everywhere it travels
POSE_ID_MAX_LENGTH = 128

# Ids are safe unquoted in annotation JSON, Label Studio label aliases, YAML class maps, model contracts, CLI
# arguments, and API paths. The leading character excludes '.' and '-' so an id is never read as a relative path
# or an option
POSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")

# The namespace Factory's labels occupy inside Label Studio. A generated labeling configuration stores this
# prefix followed by the id as a label's ``alias``, which is what Label Studio persists in a region and reports
# in an export. It makes a Factory pose project self-describing, so the fork's ownership guard recognizes one
# from its stored configuration alone. No id may start with it, in any case variant, which is what makes
# stripping it unambiguous
POSE_LABEL_ALIAS_PREFIX = "__fsp_"


def pose_label_alias(identifier: str) -> str:
    """The value Label Studio stores for the label an id names."""
    return f"{POSE_LABEL_ALIAS_PREFIX}{identifier}"


def pose_id_from_label_alias(alias: str) -> str:
    """The id a stored Label Studio value names.

    A value outside the namespace is returned unchanged rather than rejected: it is an observation of something
    the schema does not declare, and the observation checks name it against the annotation it came from.
    """
    if alias.startswith(POSE_LABEL_ALIAS_PREFIX):
        return alias[len(POSE_LABEL_ALIAS_PREFIX) :]
    return alias


# Enough digest to separate names that share a stem, short enough to leave the stem readable
_DIGEST_LENGTH = 12

# Used when a source name contains no character a stem can keep
_FALLBACK_STEM = "label"

# The longest a display name may be, counted in code points
POSE_DISPLAY_NAME_MAX_LENGTH = 128

# ECMAScript's `String.prototype.trim` set, stated here rather than relying on `str.strip()`: Python also strips
# U+001C-U+001F and U+0085 and does not strip U+FEFF, and an id derived from a source name is a persisted value
# that must not depend on which language produced it
_TRIM_CHARS = (
    "\t\n\v\f\r\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def is_pose_id(value: str) -> bool:
    """Whether ``value`` is well formed as an id. Says nothing about whether anything declares it."""
    return POSE_ID_PATTERN.match(value) is not None


def is_reserved_pose_id(value: str) -> bool:
    """Whether ``value`` claims the Label Studio alias namespace, in any case variant."""
    return value.lower().startswith(POSE_LABEL_ALIAS_PREFIX)


def derive_pose_id(source_name: str) -> str:
    """The id a source name derives, deterministically and independently of any other import.

    A name already usable as an id is kept as it stands, which is what keeps ids readable in the places they end
    up in front of someone: class maps, ``kpt_names``, training logs, and the labels an inference payload
    carries. Anything else becomes ``<stem>_<digest>``: the stem keeps the id diagnosable at a glance, and the
    digest of the whole name is what carries identity, so ``left eye`` and ``left-eye`` stay distinct rather
    than collapsing onto one concept in two datasets. The stem is ASCII-folded and lowercased purely for
    readability, and is never the thing being compared.

    No derived id carries the alias prefix, which is what lets a stored Label Studio value be read back by
    stripping it.
    """
    source = unicodedata.normalize("NFC", source_name).strip(_TRIM_CHARS)
    if is_pose_id(source) and not is_reserved_pose_id(source):
        return source
    # ASCII case folding only: locale-independent, and identical in every language that has to reproduce this
    stem = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower() or _FALLBACK_STEM
    room = POSE_ID_MAX_LENGTH - 1 - _DIGEST_LENGTH
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{stem[:room].rstrip('_')}_{digest}"


def normalize_pose_display_name(name: str) -> str:
    """The form a display name is stored and compared in."""
    return unicodedata.normalize("NFC", name).strip(_TRIM_CHARS)


def _is_disallowed_in_display_name(code: int) -> bool:
    """Code points a display name may not contain.

    Beyond the obvious controls this covers what XML 1.0 excludes from its ``Char`` production, because a name
    reaches the annotator through a generated labeling configuration, and the directional overrides and
    isolates, because a name that silently reorders what surrounds it misrepresents the interface it appears in.
    """
    return (
        code <= 0x1F  # C0 controls, which includes tab, carriage return, and line feed
        or code == 0x7F  # delete
        or 0x80 <= code <= 0x9F  # C1 controls
        or 0xD800 <= code <= 0xDFFF  # an unpaired surrogate
        or code in (0x2028, 0x2029)  # line and paragraph separators
        or 0x202A <= code <= 0x202E  # bidirectional embeddings and overrides
        or 0x2066 <= code <= 0x2069  # bidirectional isolates
        or code in (0xFFFE, 0xFFFF)
    )


def collect_pose_display_name_issues(name: str, where: str) -> List[str]:
    """Report every reason ``name`` is unusable as a display name.

    Display names are ordinary text in any script. Two rules are not about text at all. A name is a single line
    because it labels a control. And it may not contain ``$``, because Label Studio resolves a label's value
    through task-data substitution before rendering it, and there is no escape for it.
    """
    issues: List[str] = []
    normalized = normalize_pose_display_name(name)

    if not normalized:
        return [f"{where} is empty"]
    if len(normalized) > POSE_DISPLAY_NAME_MAX_LENGTH:
        issues.append(f"{where} is longer than {POSE_DISPLAY_NAME_MAX_LENGTH} characters")
    if "$" in normalized:
        issues.append(f"{where} contains '$', which Label Studio reads as a task-data substitution")
    if any(_is_disallowed_in_display_name(ord(character)) for character in normalized):
        issues.append(f"{where} contains a control or formatting character that cannot appear in a label")

    return issues


def describe_pose_label(display_name: str, identifier: str) -> str:
    """How a class or keypoint is named in text a user reads: the name first, with the id when they differ."""
    if display_name == identifier:
        return f"'{identifier}'"
    return f"'{display_name}' ({identifier})"
