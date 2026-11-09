"""Cancellation shared by long operations (downloads, fact previews, factory runs) and the Studio job manager.

Engine modules raise ``OperationCancelled`` when a cancel was requested; the Studio's job manager reports such a job
as ``cancelled`` (not ``failed``). Kept in its own module so engine code never imports the Studio.
"""

from __future__ import annotations


class OperationCancelled(Exception):
    """The user asked to stop; whatever the operation would have changed was not applied."""
