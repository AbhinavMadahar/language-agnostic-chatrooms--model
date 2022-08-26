#!/usr/bin/env python

"""
Run the machine translation engine.
"""

import re
import sys


print('WARNING: the machine translation engine is not yet implemented. For dummy purposes, all translations are identity.', file=sys.stderr)

print('Machine translation engine ready for input', file=sys.stderr)
try:
    while True:
        translation_request = input('')
        if not re.match(r'..->..: .+', translation_request):
            print(f'ERROR: {translation_request} does not match format (..->..: .+)', file=sys.stderr)
            continue

        in_lang, out_lang, text = translation_request[:2], translation_request[4:6], translation_request[8:]
        print(text)
except EOFError:
    pass