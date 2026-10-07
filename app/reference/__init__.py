"""Reference enrollment, prepared storage, and pose-selection boundaries."""

from app.reference.library import ReferenceLibraryStore
from app.reference.model import INITIAL_REFERENCE_SLOTS, ReferenceMetadata, ReferenceSlot
from app.reference.session import ReferenceLibrarySession, ReferencePersistenceError

__all__ = [
    "INITIAL_REFERENCE_SLOTS",
    "ReferenceLibraryStore",
    "ReferenceLibrarySession",
    "ReferenceMetadata",
    "ReferencePersistenceError",
    "ReferenceSlot",
]
