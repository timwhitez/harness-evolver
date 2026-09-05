"""Serialized publication with explicit committed/indeterminate outcomes.

This is NOT inode compare-and-swap against uncooperative same-UID processes.
Cooperating writers serialize on the parent directory's flock. An unexpected
post-exchange identity/content change is indeterminate: retain displaced data
and a durable intent, never exchange back over a later concurrent update.
An intent left by a crash/error blocks automatic retries until reconciliation.
"""
from __future__ import annotations

import inspect

from harness.tools.descriptor_open import embedded_opener_source, open_readonly_checked


def publish_bytes(parent: int, name: str, payload: bytes, rename_at2,
                  *, mode: int | None = None, expected_identity=None,
                  expected_sha256: str | None = None, expected_missing: bool = False,
                  lock_timeout: float = 5.0) -> dict:
    """Publish on a local Linux filesystem; return facts, not inferred rollback.

    Parent and all transaction entries must be controlled by cooperating writers.
    Advisory locking cannot contain arbitrary same-UID directory mutation.
    Readers may observe complete candidate bytes at exchange. A post-exchange
    failure is therefore never described as an untouched-target rejection.
    """
    import fcntl
    import hashlib
    import json
    import math
    import os
    import secrets
    import stat
    import time

    if not isinstance(name, str) or name in ('', '.', '..') or '/' in name:
        raise ValueError('publication requires one target path component')
    if not isinstance(payload, bytes):
        raise TypeError('publication payload must be bytes')
    if expected_missing and (expected_identity is not None or expected_sha256 is not None):
        raise ValueError('missing and existing expectations conflict')
    if isinstance(lock_timeout, bool) or not math.isfinite(lock_timeout) or lock_timeout <= 0:
        raise ValueError('lock_timeout must be finite and positive')
    journal = '.hl-publish-' + hashlib.sha256(os.fsencode(name)).hexdigest() + '.intent'
    temporary = '.hl-publish-' + secrets.token_hex(16) + '.data'
    outcome = dict(publication_protocol='hl-publication-v1',
                   publication_state='not_published', atomic_replace=False,
                   directory_fsync=False, durability_warning=False,
                   cleanup_warning=False, no_auto_retry=False,
                   recovery_entries=[], publication_error='')
    locked = claimed = temporary_exists = exchanged = published = False
    fd = None

    def retained(entry):
        if entry not in outcome['recovery_entries']:
            outcome['recovery_entries'].append(entry)

    def sync_directory():
        try:
            os.fsync(parent)
            outcome['directory_fsync'] = True
            return True
        except OSError:
            outcome['directory_fsync'] = False
            outcome['durability_warning'] = True
            return False

    def validate_entry(entry, observed=None):
        metadata = os.stat(entry, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError('Refusing symlink target')
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError('Refusing non-regular overwrite target')
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        required = expected_identity if observed is None else observed
        if required is not None and identity != tuple(required):
            raise OSError('target changed identity before publication validation')
        if expected_identity is not None or expected_sha256 is not None:
            descriptor, metadata = open_readonly_checked(parent, entry, unique=True)
            try:
                actual = (int(metadata.st_dev), int(metadata.st_ino))
                if actual != identity:
                    raise OSError('target changed during descriptor acquisition')
                if expected_sha256 is not None:
                    digest = hashlib.sha256()
                    while True:
                        chunk = os.read(descriptor, 65_536)
                        if not chunk:
                            break
                        digest.update(chunk)
                    if digest.hexdigest() != expected_sha256:
                        raise OSError('target content changed before publication validation')
            finally:
                os.close(descriptor)
        return metadata

    try:
        deadline = time.monotonic() + lock_timeout
        while True:
            try:
                fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise OSError('publication directory is busy')
                time.sleep(.01)
        # A previous claim is not an expiring lock: its publication may already
        # be visible. Never clear it simply because its process no longer lives.
        try:
            os.stat(journal, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            retained(journal)
            outcome['no_auto_retry'] = True
            raise OSError('unreconciled publication intent exists')
        try:
            current = validate_entry(name)
        except FileNotFoundError:
            if expected_identity is not None or expected_sha256 is not None:
                raise OSError('expected target disappeared before publication')
            current = None
        if expected_missing and current is not None:
            raise OSError('target appeared before publication')
        observed = None if current is None else (int(current.st_dev), int(current.st_ino))
        file_mode = mode if mode is not None else (None if current is None else current.st_mode)
        intent = dict(schema='hl-publication-v1', target=name, temporary=temporary,
                      observed_identity=observed,
                      candidate_sha256=hashlib.sha256(payload).hexdigest())
        intent_fd = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                            os.O_CLOEXEC | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        claimed = True
        try:
            with os.fdopen(intent_fd, 'wb', closefd=False) as stream:
                stream.write(json.dumps(intent, allow_nan=False).encode('utf-8') + b'\n')
                stream.flush()
                os.fsync(intent_fd)
        finally:
            os.close(intent_fd)
        # Persist the intent before any publication syscall.
        os.fsync(parent)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_CLOEXEC | os.O_NOFOLLOW, 0o666, dir_fd=parent)
        temporary_exists = True
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            if file_mode is not None:
                os.fchmod(fd, stat.S_IMODE(file_mode) & 0o777)
            os.fsync(fd)
        if current is None:
            rename_at2(parent, temporary, name, 1)  # RENAME_NOREPLACE
            temporary_exists = False
            published = True
        else:
            rename_at2(parent, temporary, name, 2)  # RENAME_EXCHANGE
            exchanged = True
            # No rollback: it cannot be an atomic inode-CAS and could erase a
            # second concurrent writer's update. Keep both evidence and intent.
            validate_entry(temporary, observed)
            published = True
        outcome['publication_state'] = 'published'
        outcome['atomic_replace'] = True
        outcome['no_auto_retry'] = True
        if not sync_directory():
            if temporary_exists:
                retained(temporary)
            retained(journal)
            return outcome
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=parent)
                temporary_exists = False
            except OSError as exc:
                outcome['cleanup_warning'] = True
                outcome['publication_error'] = str(exc)
                retained(temporary)
                retained(journal)
                return outcome
        if not sync_directory():
            retained(journal)
            return outcome
        try:
            os.unlink(journal, dir_fd=parent)
            claimed = False
        except OSError as exc:
            outcome['cleanup_warning'] = True
            outcome['publication_error'] = str(exc)
            retained(journal)
            return outcome
        sync_directory()
        return outcome
    except (OSError, ValueError) as exc:
        outcome['publication_error'] = str(exc)
        if exchanged or published:
            outcome['publication_state'] = 'indeterminate'
            outcome['atomic_replace'] = None
            outcome['no_auto_retry'] = True
            if temporary_exists:
                retained(temporary)
            if claimed:
                retained(journal)
            sync_directory()
        return outcome
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                outcome['cleanup_warning'] = True
                outcome['publication_error'] = str(exc)
        if not exchanged and not published:
            # Only remove our own pre-publication candidate. Never remove an
            # old/unknown inode displaced by an exchange, even after an error.
            if temporary_exists:
                try:
                    os.unlink(temporary, dir_fd=parent)
                    temporary_exists = False
                except OSError:
                    retained(temporary)
                    outcome['cleanup_warning'] = True
            if claimed and not temporary_exists:
                try:
                    os.unlink(journal, dir_fd=parent)
                    claimed = False
                except OSError:
                    retained(journal)
                    outcome['cleanup_warning'] = True
            if claimed:
                retained(journal)
                outcome['no_auto_retry'] = True
        if locked:
            try:
                fcntl.flock(parent, fcntl.LOCK_UN)
            except OSError as exc:
                outcome['cleanup_warning'] = True
                outcome['publication_error'] = str(exc)


def embedded_publication_source(*, include_opener: bool = True) -> str:
    """Embed the tested core, without requiring harness imports in a container."""
    return (embedded_opener_source() if include_opener else '') + inspect.getsource(publish_bytes) + '\n'
