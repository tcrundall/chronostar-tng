Frequently Asked Questions
==========================

Error::

    ReferenceError: underlying object has vanished

I believe this is to do with Numba compiling being interrupted and corrupting.
Delete ``chronostar/utils/__pycache__`` to resolve.
